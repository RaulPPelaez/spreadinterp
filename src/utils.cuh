#pragma once

#include <uammd.cuh>
#include "misc/IBM.cuh"
#include "misc/IBM_kernels.cuh"

using namespace uammd;
using real = uammd::real;
using Kernel = IBM_kernels::Peskin::threePoint;

// Spreads a group of particles onto a grid.
// The domain is such that some location r is r\in +-L
// auto spreadParticles(thrust::device_vector<real3> &pos,
//                      thrust::device_vector<complex> &values, int3 n, real sigma,
//                      int supp, real3 L) {
//   L.z *= 2;
//   auto h = L / make_real3(n);
//   auto kernel = std::make_shared<Gaussian>(sigma, h.x, L.z, n.z, supp);
//   using Grid = chebyshev::doublyperiodic::Grid;
//   Grid grid(Box(L), n);
//   IBM<Gaussian, Grid> ibm(kernel, grid);
//   auto pos_ptr = thrust::raw_pointer_cast(pos.data());
//   auto values_ptr = thrust::raw_pointer_cast(values.data());
//   thrust::device_vector<complex> d_fr(n.x * n.y * n.z);
//   thrust::fill(d_fr.begin(), d_fr.end(), complex());
//   auto fr_ptr = thrust::raw_pointer_cast(d_fr.data());
//   ibm.spread(pos_ptr, (real2 *)values_ptr, (real2 *)fr_ptr, pos.size());
//   std::vector<complex> fr(n.x * n.y * n.z, complex());
//   thrust::copy(d_fr.begin(), d_fr.end(), fr.begin());
//   return fr;
// }

// Interpolates a discrete field into the locations of a group of particles.
// The domain is such that some location r is r\in +-L
auto interpolateField(auto &pos,
                      auto &field, real3 L, int3 n) {
  real cellSize = L.x / n.x;
  auto kernel = std::make_shared<Kernel>(cellSize);
  Grid grid(Box(L), n);
  IBM<Kernel, Grid> ibm(kernel, grid);
  thrust::device_vector<real3> d_values(pos.size());
  thrust::fill(d_values.begin(), d_values.end(), real3{});
  // auto pos_ptr = thrust::raw_pointer_cast(pos.data());
  auto values_ptr = thrust::raw_pointer_cast(d_values.data());
  //auto fr_ptr = thrust::raw_pointer_cast(field.data());
  ibm.gather(pos.data(), values_ptr, field.data(), int(pos.size()));
  return d_values;
}
