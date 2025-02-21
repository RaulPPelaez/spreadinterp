#include "utils.cuh"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

namespace nb = nanobind;
using namespace nb::literals;
using uammd::real;

using pyarray_c =
    nb::ndarray<real, nb::c_contig, nb::shape<-1>, nb::device::cuda>;
using pyarray3_c =
    nb::ndarray<real, nb::c_contig, nb::shape<-1, 3>, nb::device::cuda>;
using pyarray_field_c =
    nb::ndarray<real, nb::c_contig, nb::shape<-1, -1, -1>, nb::device::cuda>;
using pyarray3f = nb::ndarray<real, nb::shape<3>, nb::device::cpu>;
using pyarray3i = nb::ndarray<int, nb::shape<3>, nb::device::cpu>;

using KernelBase = IBM_kernels::Peskin::threePoint;
struct Kernel {
  static constexpr int support = 3;
  Kernel(real3 h) : m_phiX(h.x), m_phiY(h.y), m_phiZ(h.z) {}

  __host__ __device__ real phiX(real rr, real3 pos = real3()) const {
    return m_phiX.phi(rr, pos);
  }

  __host__ __device__ real phiY(real rr, real3 pos = real3()) const {
    return m_phiY.phi(rr, pos);
  }

  __host__ __device__ real phiZ(real rr, real3 pos = real3()) const {
    return m_phiZ.phi(rr, pos);
  }
private:
  KernelBase m_phiX, m_phiY, m_phiZ;
};

struct LinearIndex3D {
  LinearIndex3D(int nx, int ny, int nz) : nx(nx), ny(ny), nz(nz) {}

  inline __device__ __host__ int operator()(int3 c) const {
    return this->operator()(c.x, c.y, c.z);
  }

  inline __device__ __host__ int operator()(int i, int j, int k) const {
    return k + nz * (j + ny * i);
  }

private:
  const int nx, ny, nz;
};

void cudaCheckError() {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    throw std::runtime_error("CUDA error: " +
                             std::string(cudaGetErrorString(err)));
  }
}

void interpolateField_wrapper(pyarray3_c ipos, pyarray_field_c ifield,
                              pyarray_c iquantity, pyarray3f Li) {
  if (ipos.shape(0) != iquantity.shape(0)) {
    throw std::runtime_error("Quantity shape does not match pos");
  }
  auto ni = ifield.shape_ptr();
  real3 L = {Li.view()(0), Li.view()(1), Li.view()(2)};
  int3 n = {int(ni[0]), int(ni[1]), int(ni[2])};
  real3 cellSize = L / make_real3(n);
  auto kernel = std::make_shared<Kernel>(cellSize);
  Grid grid(Box(L), n);
  IBM<Kernel, Grid, LinearIndex3D> ibm(kernel, grid);
  ibm.gather(reinterpret_cast<real3 *>(ipos.data()),
             reinterpret_cast<real *>(iquantity.data()),
             reinterpret_cast<real *>(ifield.data()), int(ipos.shape(0)));
  cudaCheckError();
}

auto spreadParticles_wrapper(pyarray3_c ipos, pyarray_c iquantity,
                             pyarray_field_c ifield, pyarray3f Li,
                             pyarray3i ni) {
  if (iquantity.shape(0) != ipos.shape(0)) {
    throw std::runtime_error("Quantity shape does not match pos");
  }
  real3 L = {Li.view()(0), Li.view()(1), Li.view()(2)};
  int3 n = {ni.view()(0), ni.view()(1), ni.view()(2)};
  real3 cellSize = L / make_real3(n);
  auto kernel = std::make_shared<Kernel>(cellSize);
  Grid grid(Box(L), n);
  IBM<Kernel, Grid, LinearIndex3D> ibm(kernel, grid);

  ibm.spread(reinterpret_cast<real3 *>(ipos.data()),
             reinterpret_cast<real *>(iquantity.data()),
             reinterpret_cast<real *>(ifield.data()), int(ipos.shape(0)));
  cudaCheckError();
}

NB_MODULE(_spreadinterp, m) {
  m.def("interpolateField", &interpolateField_wrapper, "pos"_a, "field"_a,
        "quantity"_a.noconvert(), "L"_a);
  m.def("spreadParticles", &spreadParticles_wrapper, "pos"_a, "quantity"_a,
        "field"_a.noconvert(), "L"_a, "n"_a);
}
