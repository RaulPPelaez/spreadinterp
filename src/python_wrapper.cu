#include "utils.cuh"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <optional>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

namespace nb = nanobind;
using namespace nb::literals;
using uammd::real;

using pyarray_c =
    nb::ndarray<real, nb::c_contig, nb::shape<-1, -1>, nb::device::cuda>;
using pyarray3_c =
    nb::ndarray<real, nb::c_contig, nb::shape<-1, 3>, nb::device::cuda>;
using pyarray_field_c =
    nb::ndarray<real, nb::c_contig, nb::shape<-1, -1, -1, -1>,
                nb::device::cuda>;
using pyarray3f = nb::ndarray<real, nb::shape<3>, nb::device::cpu>;
using pyarray3i = nb::ndarray<int, nb::shape<3>, nb::device::cpu>;

namespace kernel {
enum class type { peskin3pt, gaussian };

namespace peskin3pt {
struct KernelParameters {
  // No additional parameters needed for the peskin3pt kernel
};

using KernelBase = IBM_kernels::Peskin::threePoint;

struct Kernel {
  static constexpr int support = 3;
  Kernel(real3 h, bool is2D = false)
      : m_phiX(h.x), m_phiY(h.y), m_phiZ(h.z), is2D(is2D) {}

  __host__ __device__ real phiX(real rr, real3 pos = real3()) const {
    return m_phiX.phi(rr, pos);
  }

  __host__ __device__ real phiY(real rr, real3 pos = real3()) const {
    return m_phiY.phi(rr, pos);
  }

  __host__ __device__ real phiZ(real rr, real3 pos = real3()) const {
    return is2D ? real(1.0) : m_phiZ.phi(rr, pos);
  }

private:
  KernelBase m_phiX, m_phiY, m_phiZ;
  bool is2D;
};

struct DerivativeBase {
  const real invh;
  static constexpr int support = 3;
  DerivativeBase(real h) : invh(1.0 / h) {}

  __host__ __device__ real phi(real rr, real3 pos = real3()) const {
    real r = fabs(rr) * invh;
    real sgn = (rr >= 0) ? 1.0 : -1.0;

    if (r < real(0.5)) {
      return -invh * invh * r * sgn / sqrt(real(1.0) - real(3.0) * r * r);
    } else if (r < real(1.5)) {
      return -invh * invh * (real(1.0) / real(2.0)) *
             (real(1.0) +
              (real(1.0) - r) / sqrt(real(1.0) - real(3.0) * (real(1.0) - r) *
                                                     (real(1.0) - r))) *
             sgn;
    } else {
      return 0;
    }
  }
};
} // namespace peskin3pt
namespace gaussian {
struct KernelParameters {
  real width;  // Width of the Gaussian kernel
  real cutoff; // Cutoff distance for the kernel
};
struct KernelBase {
  int support;
  real width;
  real cellSize;
  real cutoff;
  real prefactor;
  KernelBase(real cellSize, real width, real cutoff) {
    this->support = 2 * cutoff / cellSize + 1;
    this->width = width;
    this->cellSize = cellSize;
    this->cutoff = cutoff;
    this->prefactor = pow(2 * M_PI * width * width, -1.5);
  }
  __host__ __device__ real phi(real rr, real3 pos = real3()) const {
    real r = fabs(rr) / cellSize;
    if (fabs(rr) < cutoff) {
      return prefactor*exp(-r * r / (real(2.0) * width * width));
    } else {
      return 0.0;
    }
  }
};

struct Kernel {
  int support;
  bool is2D;
  KernelBase m_phiX, m_phiY, m_phiZ;
  Kernel(real3 cellSize, bool is2D, real width, real cutoff)
      : m_phiX(cellSize.x, width, cutoff), m_phiY(cellSize.y, width, cutoff),
        m_phiZ(cellSize.z, width, cutoff) {
    this->support = std::max({m_phiX.support, m_phiY.support, m_phiZ.support});
    this->is2D = is2D;
  }
  __host__ __device__ real phiX(real rr, real3 pos = real3()) const {
    return m_phiX.phi(rr, pos);
  }
  __host__ __device__ real phiY(real rr, real3 pos = real3()) const {
    return m_phiY.phi(rr, pos);
  }
  __host__ __device__ real phiZ(real rr, real3 pos = real3()) const {
    return is2D ? real(1.0) : m_phiZ.phi(rr, pos);
  }
};

struct DerivativeBase {
  int support;
  real width;
  real cellSize;
  real prefactor;
  DerivativeBase(real cellSize, real width, real cutoff) {
    this->support = 2 * cutoff / cellSize + 1;
    this->width = width;
    this->cellSize = cellSize;
    this->prefactor = pow(2 * M_PI * width * width, -1.5);
  }
  __host__ __device__ real phi(real rr, real3 pos = real3()) const {
    real r = fabs(rr) / cellSize;
    if (r < support) {
      real sgn = (rr >= 0) ? 1.0 : -1.0;
      return -prefactor*sgn * (r / (width * width)) *
             exp(-r * r / (real(2.0) * width * width));
    } else {
      return 0.0;
    }
  }
};

} // namespace gaussian
} // namespace kernel
template <typename Kernel, typename KernelDerivative> struct GradientKernel {
  int support;

  GradientKernel(Kernel phiX, Kernel phiY, Kernel phiZ, KernelDerivative dphiX,
                 KernelDerivative dphiY, KernelDerivative dphiZ,
                 bool is2D = false)
      : m_phiX(phiX), m_phiY(phiY), m_phiZ(phiZ), m_dphiX(dphiX),
        m_dphiY(dphiY), m_dphiZ(dphiZ), is2D(is2D) {
    support = std::max({m_phiX.support, m_phiY.support, m_phiZ.support,
                        m_dphiX.support, m_dphiY.support, m_dphiZ.support});
  }

  __host__ __device__ std::tuple<real, real> phiX(real r,
                                                  real3 pos = real3()) const {
    return {m_phiX.phi(r, pos), m_dphiX.phi(r, pos)};
  }

  __host__ __device__ std::tuple<real, real> phiY(real r,
                                                  real3 pos = real3()) const {
    return {m_phiY.phi(r, pos), m_dphiY.phi(r, pos)};
  }

  __host__ __device__ std::tuple<real, real> phiZ(real r,
                                                  real3 pos = real3()) const {
    return {is2D ? real(1.0) : m_phiZ.phi(r, pos),
            is2D ? real(0.0) : m_dphiZ.phi(r, pos)};
  }

private:
  Kernel m_phiX, m_phiY, m_phiZ;
  KernelDerivative m_dphiX, m_dphiY, m_dphiZ;
  bool is2D;
};

struct GradientInterpolationWeightCompute {
  template <typename T2>
  inline __device__ real3 operator()(real quantity,
                                     thrust::tuple<T2, T2, T2> kernel) const {
    auto [phiX, dphiX] = thrust::get<0>(kernel);
    auto [phiY, dphiY] = thrust::get<1>(kernel);
    auto [phiZ, dphiZ] = thrust::get<2>(kernel);
    real3 delta = {phiY * phiZ * dphiX, phiX * phiZ * dphiY,
                   phiX * phiY * dphiZ};
    return delta * quantity;
  }
};

struct GradientSpreadWeightCompute {
  template <typename T, typename T2>
  inline __device__ T operator()(thrust::tuple<real3, T> iquantity,
                                 thrust::tuple<T2, T2, T2> kernel) const {
    auto [phiX, dphiX] = thrust::get<0>(kernel);
    auto [phiY, dphiY] = thrust::get<1>(kernel);
    auto [phiZ, dphiZ] = thrust::get<2>(kernel);
    const auto quantity = thrust::get<1>(iquantity);
    const real3 direction = thrust::get<0>(iquantity);
    const real delta = phiY * phiZ * dphiX * direction.x +
                       phiX * phiZ * dphiY * direction.y +
                       phiX * phiY * dphiZ * direction.z;
    return delta * quantity;
  }
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

struct Permute {
  Permute(int nf, int i) : nf(nf), i(i) {}

  inline __device__ __host__ int operator()(int j) const { return j * nf + i; }

private:
  int nf, i;
};

void cudaCheckError() {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    throw std::runtime_error("CUDA error: " +
                             std::string(cudaGetErrorString(err)));
  }
}

void dispatchWithReal(auto &foo, pyarray_field_c ifield, pyarray_c iquantity) {
  if (iquantity.shape(1) == 1) {
    foo(reinterpret_cast<real *>(iquantity.data()),
        reinterpret_cast<real *>(ifield.data()));
  } else if (iquantity.shape(1) == 2) {
    foo(reinterpret_cast<real2 *>(iquantity.data()),
        reinterpret_cast<real2 *>(ifield.data()));
  } else if (iquantity.shape(1) == 3) {
    foo(reinterpret_cast<real3 *>(iquantity.data()),
        reinterpret_cast<real3 *>(ifield.data()));
  } else if (iquantity.shape(1) == 4) {
    foo(reinterpret_cast<real4 *>(iquantity.data()),
        reinterpret_cast<real4 *>(ifield.data()));
  } else {
    auto f_ptr = reinterpret_cast<real *>(ifield.data());
    auto q_ptr = reinterpret_cast<real *>(iquantity.data());
    int nf = ifield.shape(3);
    for (int i = 0; i < iquantity.shape(1); i++) {
      auto f_it = thrust::make_permutation_iterator(
          f_ptr, thrust::make_transform_iterator(
                     thrust::make_counting_iterator(0), Permute(nf, i)));
      auto q_it = thrust::make_permutation_iterator(
          q_ptr, thrust::make_transform_iterator(
                     thrust::make_counting_iterator(0), Permute(nf, i)));
      foo(q_it, f_it);
    }
  }
}

template <kernel::type type, typename KernelParameters>
auto createKernel(real3 L, int3 n, KernelParameters kpar) {
  const real3 cellSize = L / make_real3(n);
  if constexpr (type == kernel::type::peskin3pt) {
    return std::make_shared<kernel::peskin3pt::Kernel>(cellSize, n.z == 1);
  } else if constexpr (type == kernel::type::gaussian) {
    return std::make_shared<kernel::gaussian::Kernel>(cellSize, n.z == 1,
                                                      kpar.width, kpar.cutoff);
  } else {
    throw std::runtime_error("Unsupported kernel type");
  }
}

template <kernel::type type, typename KernelParameters>
auto createGradientKernel(real3 L, int3 n, KernelParameters kpar) {
  const real3 cellSize = L / make_real3(n);
  if constexpr (type == kernel::type::peskin3pt) {
    using Kernel = GradientKernel<kernel::peskin3pt::KernelBase,
                                  kernel::peskin3pt::DerivativeBase>;
    return std::make_shared<Kernel>(
        kernel::peskin3pt::KernelBase(cellSize.x),
        kernel::peskin3pt::KernelBase(cellSize.y),
        kernel::peskin3pt::KernelBase(cellSize.z),
        kernel::peskin3pt::DerivativeBase(cellSize.x),
        kernel::peskin3pt::DerivativeBase(cellSize.y),
        kernel::peskin3pt::DerivativeBase(cellSize.z), n.z == 1);
  } else if constexpr (type == kernel::type::gaussian) {
    using Kernel = GradientKernel<kernel::gaussian::KernelBase,
                                  kernel::gaussian::DerivativeBase>;
    return std::make_shared<Kernel>(
        kernel::gaussian::KernelBase(cellSize.x, kpar.width, kpar.cutoff),
        kernel::gaussian::KernelBase(cellSize.y, kpar.width, kpar.cutoff),
        kernel::gaussian::KernelBase(cellSize.z, kpar.width, kpar.cutoff),
        kernel::gaussian::DerivativeBase(cellSize.x, kpar.width, kpar.cutoff),
        kernel::gaussian::DerivativeBase(cellSize.y, kpar.width, kpar.cutoff),
        kernel::gaussian::DerivativeBase(cellSize.z, kpar.width, kpar.cutoff),
        n.z == 1);
  } else {
    throw std::runtime_error("Unsupported gradient kernel type");
  }
}

template <kernel::type type>
auto getKernelParameters(nb::dict kernel_descriptor) {
  if constexpr (type == kernel::type::peskin3pt) {
    if (nb::cast<std::string>(kernel_descriptor["type"]) != "peskin3pt") {
      throw std::runtime_error("Kernel type mismatch, expected peskin3pt");
    }
    return kernel::peskin3pt::KernelParameters();
  } else if constexpr (type == kernel::type::gaussian) {
    if (nb::cast<std::string>(kernel_descriptor["type"]) != "gaussian") {
      throw std::runtime_error("Kernel type mismatch, expected gaussian");
    }
    real width = nb::cast<double>(kernel_descriptor["width"]);
    real cutoff = nb::cast<double>(kernel_descriptor["cutoff"]);
    return kernel::gaussian::KernelParameters(width, cutoff);
  } else {
    throw std::runtime_error("Unsupported kernel type");
  }
}

template <typename Kernel>
void interpolateField_direct(pyarray3_c ipos, pyarray_field_c ifield,
                             pyarray_c iquantity, real3 L,
                             std::shared_ptr<Kernel> kernel) {
  const auto ni = ifield.shape_ptr();
  const int3 n = {int(ni[0]), int(ni[1]), int(ni[2])};
  Grid grid(Box(L), n);
  IBM<Kernel, Grid, LinearIndex3D> ibm(kernel, grid);
  auto gather = [&](auto q_it, auto f_it) {
    ibm.gather(reinterpret_cast<real3 *>(ipos.data()), q_it, f_it,
               int(ipos.shape(0)));
  };
  dispatchWithReal(gather, ifield, iquantity);
  cudaCheckError();
}

struct Dot {
  inline __device__ __host__ real operator()(real3 a, real3 b) const {
    return a.x * b.x + a.y * b.y + a.z * b.z;
  }
};

template <typename GradientKernel>
void interpolateField_gradient(pyarray3_c ipos, pyarray_field_c ifield,
                               pyarray_c iquantity, pyarray3_c idirection,
                               real3 L,
                               std::shared_ptr<GradientKernel> kernel) {
  if (idirection.shape(0) != ipos.shape(0)) {
    throw std::runtime_error(
        "Gradient direction must have same number of particles as pos");
  }
  const auto ni = ifield.shape_ptr();
  const int3 n = {int(ni[0]), int(ni[1]), int(ni[2])};
  Grid grid(Box(L), n);
  IBM<GradientKernel, Grid, LinearIndex3D> ibm(kernel, grid);
  auto wc = GradientInterpolationWeightCompute();
  auto qw = IBM_ns::DefaultQuadratureWeights();
  auto q_it = reinterpret_cast<real *>(iquantity.data());
  auto d_ptr = reinterpret_cast<real3 *>(idirection.data());
  thrust::device_vector<real3> qd(ipos.shape(0));
  for (int i = 0; i < ifield.shape(3); i++) {
    thrust::fill(thrust::cuda::par, qd.begin(), qd.end(), real3());
    auto f_a = thrust::make_permutation_iterator(
        reinterpret_cast<real *>(ifield.data()),
        thrust::make_transform_iterator(thrust::make_counting_iterator(0),
                                        Permute(ifield.shape(3), i)));
    ibm.gather(reinterpret_cast<real3 *>(ipos.data()), qd.data().get(), f_a, qw,
               wc, int(ipos.shape(0)));
    auto qa_it = thrust::make_permutation_iterator(
        q_it, thrust::make_transform_iterator(thrust::make_counting_iterator(0),
                                              Permute(3, i)));
    thrust::transform(thrust::cuda::par, qd.begin(), qd.end(), d_ptr, qa_it,
                      Dot());
  }

  cudaCheckError();
}
template <kernel::type type, typename KernelParameters>
void _interpolateField_wrapper(pyarray3_c ipos, pyarray_field_c ifield,
                               pyarray_c iquantity, pyarray3f Li, bool gradient,
                               std::optional<pyarray3_c> gradient_direction,
                               KernelParameters kpar) {
  if (ipos.shape(0) != iquantity.shape(0)) {
    throw std::runtime_error("Quantity shape does not match pos");
  }
  if (iquantity.shape(1) != ifield.shape(3)) {
    throw std::runtime_error("Quantity shape does not match field");
  }
  real3 L = {Li.view()(0), Li.view()(1), Li.view()(2)};
  int3 n = {int(ifield.shape(0)), int(ifield.shape(1)), int(ifield.shape(2))};
  if (!gradient) {
    auto kernel = createKernel<type>(L, n, kpar);
    interpolateField_direct(ipos, ifield, iquantity, L, kernel);
  } else {
    if (!gradient_direction.has_value()) {
      throw std::runtime_error("Gradient direction must be provided");
    }
    auto kernel = createGradientKernel<type>(L, n, kpar);
    interpolateField_gradient(ipos, ifield, iquantity,
                              gradient_direction.value(), L, kernel);
  }
}

void interpolateField_wrapper(pyarray3_c ipos, pyarray_field_c ifield,
                              pyarray_c iquantity, pyarray3f Li, bool gradient,
                              std::optional<pyarray3_c> gradient_direction,
                              std::optional<nb::dict> kernel_descriptor) {
  if (!kernel_descriptor.has_value()) {
    throw std::runtime_error(
        "Kernel descriptor must be provided for interpolateField");
  }
  std::string kernel_type =
      nb::cast<std::string>((kernel_descriptor.value())["type"]);
  if (kernel_type == "peskin3pt") {
    auto kpar =
        getKernelParameters<kernel::type::peskin3pt>(kernel_descriptor.value());
    _interpolateField_wrapper<kernel::type::peskin3pt>(
        ipos, ifield, iquantity, Li, gradient, gradient_direction, kpar);
  } else if (kernel_type == "gaussian") {
    auto kpar =
        getKernelParameters<kernel::type::gaussian>(kernel_descriptor.value());
    _interpolateField_wrapper<kernel::type::gaussian>(
        ipos, ifield, iquantity, Li, gradient, gradient_direction, kpar);
  } else {
    throw std::runtime_error("Unsupported kernel type: " + kernel_type);
  }
}

template <typename Kernel>
void spreadParticles_direct(pyarray3_c ipos, pyarray_c iquantity,
                            pyarray_field_c ifield, real3 L, int3 n,
                            std::shared_ptr<Kernel> kernel) {
  Grid grid(Box(L), n);
  IBM<Kernel, Grid, LinearIndex3D> ibm(kernel, grid);
  auto spread = [&](auto q_it, auto f_it) {
    ibm.spread(reinterpret_cast<real3 *>(ipos.data()), q_it, f_it,
               int(ipos.shape(0)));
  };
  dispatchWithReal(spread, ifield, iquantity);
  cudaCheckError();
}

template <typename GradientKernel>
void spreadParticles_gradient(pyarray3_c ipos, pyarray_c iquantity,
                              pyarray_field_c ifield, pyarray3_c idirection,
                              real3 L, int3 n,
                              std::shared_ptr<GradientKernel> kernel) {
  if (idirection.shape(0) != ipos.shape(0)) {
    throw std::runtime_error(
        "Gradient direction must have same number of particles as pos");
  }
  Grid grid(Box(L), n);
  IBM<GradientKernel, Grid, LinearIndex3D> ibm(kernel, grid);
  auto d_ptr = reinterpret_cast<real3 *>(idirection.data());
  auto wc = GradientSpreadWeightCompute();
  for (int i = 0; i < ifield.shape(3); i++) {
    auto q_it_i = reinterpret_cast<real *>(iquantity.data());
    auto q_it = thrust::make_permutation_iterator(
        q_it_i,
        thrust::make_transform_iterator(thrust::make_counting_iterator(0),
                                        Permute(ifield.shape(3), i)));
    auto dq_it = thrust::make_zip_iterator(thrust::make_tuple(d_ptr, q_it));
    // auto f_it = reinterpret_cast<real *>(ifield.data());
    auto f_it = thrust::make_permutation_iterator(
        reinterpret_cast<real *>(ifield.data()),
        thrust::make_transform_iterator(thrust::make_counting_iterator(0),
                                        Permute(ifield.shape(3), i)));
    ibm.spread(reinterpret_cast<real3 *>(ipos.data()), dq_it, f_it, wc,
               int(ipos.shape(0)));
    cudaCheckError();
  }
}

template <kernel::type type, typename KernelParameters>
void _spreadParticles_wrapper(pyarray3_c ipos, pyarray_c iquantity,
                              pyarray_field_c ifield, pyarray3f Li,
                              pyarray3i ni, bool gradient,
                              std::optional<pyarray3_c> gradient_direction,
                              KernelParameters kpar) {
  if (ipos.shape(0) != iquantity.shape(0)) {
    throw std::runtime_error("Quantity shape does not match pos");
  }
  if (iquantity.shape(1) != ifield.shape(3)) {
    throw std::runtime_error("Quantity shape does not match field");
  }
  real3 L = {Li.view()(0), Li.view()(1), Li.view()(2)};
  int3 n = {ni.view()(0), ni.view()(1), ni.view()(2)};
  if (!gradient) {
    auto kernel = createKernel<type>(L, n, kpar);
    spreadParticles_direct(ipos, iquantity, ifield, L, n, kernel);
  } else {
    auto kernel = createGradientKernel<type>(L, n, kpar);
    spreadParticles_gradient(ipos, iquantity, ifield,
                             gradient_direction.value(), L, n, kernel);
  }
}

void spreadParticles_wrapper(pyarray3_c ipos, pyarray_c iquantity,
                             pyarray_field_c ifield, pyarray3f Li, pyarray3i ni,
                             bool gradient,
                             std::optional<pyarray3_c> gradient_direction,
                             std::optional<nb::dict> kernel_descriptor) {
  if (!kernel_descriptor.has_value()) {
    throw std::runtime_error(
        "Kernel descriptor must be provided for spreadParticles");
  }
  std::string kernel_type =
      nb::cast<std::string>((kernel_descriptor.value())["type"]);
  if (kernel_type == "peskin3pt") {
    auto kpar =
        getKernelParameters<kernel::type::peskin3pt>(kernel_descriptor.value());
    _spreadParticles_wrapper<kernel::type::peskin3pt>(
        ipos, iquantity, ifield, Li, ni, gradient, gradient_direction, kpar);
  } else if (kernel_type == "gaussian") {
    auto kpar =
        getKernelParameters<kernel::type::gaussian>(kernel_descriptor.value());
    _spreadParticles_wrapper<kernel::type::gaussian>(
        ipos, iquantity, ifield, Li, ni, gradient, gradient_direction, kpar);
  } else {
    throw std::runtime_error("Unsupported kernel type: " + kernel_type);
  }
}

NB_MODULE(_spreadinterp, m) {
  m.def("interpolateField", &interpolateField_wrapper, "pos"_a, "field"_a,
        "quantity"_a.noconvert(), "L"_a, "gradient"_a = false,
        "gradient_direction"_a = nb::none(),
        "kernel_descriptor"_a = nb::none());
  m.def("spreadParticles", &spreadParticles_wrapper, "pos"_a, "quantity"_a,
        "field"_a.noconvert(), "L"_a, "n"_a, "gradient"_a = false,
        "gradient_direction"_a = nb::none(),
        "kernel_descriptor"_a = nb::none());
}
