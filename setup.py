import os
import sys
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

try:
    version = (
        subprocess.check_output(["git", "describe", "--abbrev=0", "--tags"])
        .strip()
        .decode("utf-8")
    )
except:
    print("Failed to retrieve the current version, defaulting to 0")
    version = "0"


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extra_cmake_args = os.environ.get("CMAKE_ARGS", "")
        cmake_args = [
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_INSTALL_PREFIX=" + sys.prefix,
        ]
        cmake_args += extra_cmake_args.split()
        build_args = ["--config", "Release"]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", ".", "--parallel", "--target", "install"] + build_args,
            cwd=self.build_temp,
        )


setup(
    name="spreadinterp",
    version=version,
    author="Raul P. Pelaez",
    description="Python bindings for interpolate and spread from UAMMD",
    ext_modules=[CMakeExtension("spreadinterp")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
