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

void interpolateField_wrapper(pyarray3_c ipos, pyarray_field_c ifield,
                              pyarray_c iquantity, pyarray3f Li) {
  auto ni = ifield.shape_ptr();
  real3 L = {Li.view()(0), Li.view()(1), Li.view()(2)};
  int3 n = {int(ni[0]), int(ni[1]), int(ni[2])};
  real cellSize = L.x / n.x;
  auto kernel = std::make_shared<Kernel>(cellSize);
  Grid grid(Box(L), n);
  IBM<Kernel, Grid> ibm(kernel, grid);
  ibm.gather(ipos.data(), iquantity.data(), ifield.data(), int(ipos.shape(0)));
}

auto spreadParticles_wrapper(pyarray3_c ipos, pyarray_c iquantity,
                             pyarray_field_c ifield, pyarray3f Li,
                             pyarray3i ni) {
  if (iquantity.shape(0) != ipos.shape(0)) {
    throw std::runtime_error("Quantity shape does not match pos");
  }
  real3 L = {Li.view()(0), Li.view()(1), Li.view()(2)};
  int3 n = {ni.view()(0), ni.view()(1), ni.view()(2)};
  real cellSize = L.x / n.x;
  auto kernel = std::make_shared<Kernel>(cellSize);
  Grid grid(Box(L), n);
  IBM<Kernel, Grid> ibm(kernel, grid);

  ibm.spread(ipos.data(), iquantity.data(), ifield.data(), int(ipos.shape(0)));
}

// constexpr const char *interpolate_docstring = R" ";

// constexpr const char *spread_docstring = R"()";

NB_MODULE(_spreadinterp, m) {
  m.def("interpolateField", &interpolateField_wrapper, "pos"_a, "field"_a,
        "quantity"_a.noconvert(), "L"_a);
  m.def("spreadParticles", &spreadParticles_wrapper, "pos"_a, "quantity"_a,
        "field"_a.noconvert(), "L"_a, "n"_a);
}
