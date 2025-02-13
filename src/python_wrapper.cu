#include "spreadinterp.cuh"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

namespace nb = nanobind;
using namespace nb::literals;
using uammd::real;

using pyarray_c =
    nb::ndarray<real, nb::c_contig, nb::shape<-1, 3>, nb::device::cuda>;
using pyarray_field_c =
    nb::ndarray<real, nb::c_contig, nb::shape<-1, -1, -1, 3>, nb::device::cuda>;
using pyarray3f = nb::ndarray<real, nb::shape<3>, nb::device::cpu>;
using pyarray3i = nb::ndarray<int, nb::shape<3>, nb::device::cpu>;

auto interpolateField_wrapper(const pyarray_c &ipos,
                              const pyarray_field_c &ifield,
                              const pyarray3f &Li, const pyarray3i &ni) {
  if (ifield.shape(0) != ni.view()(0) || ifield.shape(1) != ni.view()(1) ||
      ifield.shape(2) != ni.view()(2)) {
    throw std::runtime_error("Field shape does not match ni");
  }
  real3 L = {Li.view()(0), Li.view()(1), Li.view()(2)};
  int3 n = {ni.view()(0), ni.view()(1), ni.view()(2)};
  struct Capsule {
    thrust::device_vector<real3> v;
  };
  Capsule *c = new Capsule();
  c->v = std::move(interpolateField(ipos, ifield, L, n));
  nb::capsule deleter(c, [](void *p) noexcept { delete (Capsule *)p; });
  return nb::ndarray<real, nb::device::cuda, nb::cupy>(
      c->v.data().get(), {ipos.shape(0), ipos.shape(1)}, deleter);
}

constexpr const char *interpolate_docstring = R"(
Interpolate a field defined on a grid to a set of points.
Field is assumed to be defined on a regular grid with periodic boundary conditions defined from  [-L/2 to L/2) in each direction.
Positions and field must be in cuda memory.

Parameters
----------
pos : ndarray
    The positions of the points to interpolate to.
field : ndarray
    The field to interpolate.
L : ndarray
    The box size.
n : ndarray
    The number of grid points in each direction.

Returns
-------

interpolated_field : ndarray
    The interpolated field at the points.
)";


NB_MODULE(spreadinterp, m) {
  m.def("interpolateField", &interpolateField_wrapper, "pos"_a, "field"_a,
        "L"_a, "n"_a, interpolate_docstring);
}
