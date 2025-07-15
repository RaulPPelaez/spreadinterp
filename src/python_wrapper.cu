#include "utils.cuh"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
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

struct threePointDerivative {
  const real invh;
  static constexpr int support = 3;
  threePointDerivative(real h) : invh(1.0 / h) {}

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

struct GradientKernel {
  static constexpr int support = 3;

  GradientKernel(real3 h, bool is2D)
      : m_phiX(h.x), m_phiY(h.y), m_phiZ(h.z), m_dphiX(h.x), m_dphiY(h.y),
        m_dphiZ(h.z), is2D(is2D) {}

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
  KernelBase m_phiX, m_phiY, m_phiZ;
  threePointDerivative m_dphiX, m_dphiY, m_dphiZ;
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
  template <class T2>
  inline __device__ real3 operator()(thrust::tuple<real3, real3> iquantity,
                                     thrust::tuple<T2, T2, T2> kernel) const {
    auto [phiX, dphiX] = thrust::get<0>(kernel);
    auto [phiY, dphiY] = thrust::get<1>(kernel);
    auto [phiZ, dphiZ] = thrust::get<2>(kernel);
    real3 quantity = thrust::get<1>(iquantity);
    real3 direction = thrust::get<0>(iquantity);
    real delta = phiY * phiZ * dphiX * direction.x +
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

void interpolateField_direct(pyarray3_c ipos, pyarray_field_c ifield,
                             pyarray_c iquantity, real3 L) {
  const auto ni = ifield.shape_ptr();
  const int3 n = {int(ni[0]), int(ni[1]), int(ni[2])};
  const real3 cellSize = L / make_real3(n);
  auto kernel = std::make_shared<Kernel>(cellSize, n.z == 1);
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

void interpolateField_gradient(pyarray3_c ipos, pyarray_field_c ifield,
                               pyarray_c iquantity, pyarray3_c idirection,
                               real3 L) {
  if (iquantity.shape(1) != 3) {
    throw std::runtime_error("Quantity must be 3D");
  }
  if (idirection.shape(0) != ipos.shape(0)) {
    throw std::runtime_error(
        "Gradient direction must have same number of particles as pos");
  }
  const auto ni = ifield.shape_ptr();
  const int3 n = {int(ni[0]), int(ni[1]), int(ni[2])};
  const real3 cellSize = L / make_real3(n);
  auto kernel = std::make_shared<GradientKernel>(cellSize, n.z == 1);
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
void interpolateField_wrapper(pyarray3_c ipos, pyarray_field_c ifield,
                              pyarray_c iquantity, pyarray3f Li, bool gradient,
                              std::optional<pyarray3_c> gradient_direction) {
  if (ipos.shape(0) != iquantity.shape(0)) {
    throw std::runtime_error("Quantity shape does not match pos");
  }
  if (iquantity.shape(1) != ifield.shape(3)) {
    throw std::runtime_error("Quantity shape does not match field");
  }
  real3 L = {Li.view()(0), Li.view()(1), Li.view()(2)};
  if (!gradient) {
    interpolateField_direct(ipos, ifield, iquantity, L);
  } else {
    if (!gradient_direction.has_value()) {
      throw std::runtime_error("Gradient direction must be provided");
    }
    interpolateField_gradient(ipos, ifield, iquantity,
                              gradient_direction.value(), L);
  }
}

void spreadParticles_direct(pyarray3_c ipos, pyarray_c iquantity,
                            pyarray_field_c ifield, real3 L, int3 n) {
  real3 cellSize = L / make_real3(n);
  auto kernel = std::make_shared<Kernel>(cellSize, n.z == 1);
  Grid grid(Box(L), n);
  IBM<Kernel, Grid, LinearIndex3D> ibm(kernel, grid);
  auto spread = [&](auto q_it, auto f_it) {
    ibm.spread(reinterpret_cast<real3 *>(ipos.data()), q_it, f_it,
               int(ipos.shape(0)));
  };
  dispatchWithReal(spread, ifield, iquantity);
  cudaCheckError();
}

void spreadParticles_gradient(pyarray3_c ipos, pyarray_c iquantity,
                              pyarray_field_c ifield, pyarray3_c idirection,
                              real3 L, int3 n) {
  if (iquantity.shape(1) != 3) {
    throw std::runtime_error("Quantity must be 3D");
  }
  if (idirection.shape(0) != ipos.shape(0)) {
    throw std::runtime_error(
        "Gradient direction must have same number of particles as pos");
  }
  real3 cellSize = L / make_real3(n);
  auto kernel = std::make_shared<GradientKernel>(cellSize, n.z == 1);
  Grid grid(Box(L), n);
  IBM<GradientKernel, Grid, LinearIndex3D> ibm(kernel, grid);
  auto wc = GradientSpreadWeightCompute();
  auto q_it = reinterpret_cast<real3 *>(iquantity.data());
  auto d_ptr = reinterpret_cast<real3 *>(idirection.data());
  auto dq_it = thrust::make_zip_iterator(thrust::make_tuple(d_ptr, q_it));
  auto f_it = reinterpret_cast<real3 *>(ifield.data());
  ibm.spread(reinterpret_cast<real3 *>(ipos.data()), dq_it, f_it, wc,
             int(ipos.shape(0)));
  cudaCheckError();
}

void spreadParticles_wrapper(pyarray3_c ipos, pyarray_c iquantity,
                             pyarray_field_c ifield, pyarray3f Li, pyarray3i ni,
                             bool gradient,
                             std::optional<pyarray3_c> gradient_direction) {
  if (ipos.shape(0) != iquantity.shape(0)) {
    throw std::runtime_error("Quantity shape does not match pos");
  }
  if (iquantity.shape(1) != ifield.shape(3)) {
    throw std::runtime_error("Quantity shape does not match field");
  }
  real3 L = {Li.view()(0), Li.view()(1), Li.view()(2)};
  int3 n = {ni.view()(0), ni.view()(1), ni.view()(2)};
  if (!gradient)
    spreadParticles_direct(ipos, iquantity, ifield, L, n);
  else
    spreadParticles_gradient(ipos, iquantity, ifield,
                             gradient_direction.value(), L, n);
}

NB_MODULE(_spreadinterp, m) {
  m.def("interpolateField", &interpolateField_wrapper, "pos"_a, "field"_a,
        "quantity"_a.noconvert(), "L"_a, "gradient"_a = false,
        "gradient_direction"_a = nb::none());
  m.def("spreadParticles", &spreadParticles_wrapper, "pos"_a, "quantity"_a,
        "field"_a.noconvert(), "L"_a, "n"_a, "gradient"_a = false,
        "gradient_direction"_a = nb::none());
}
