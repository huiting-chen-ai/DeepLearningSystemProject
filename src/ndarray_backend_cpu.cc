#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <iostream>
#include <stdexcept>

#include <complex>
#include <cstddef>

namespace needle {
namespace cpu {

#define ALIGNMENT 256
#define TILE 8
template <typename scalar_t>
struct ScalarTraits {
    using type = scalar_t;
    static constexpr size_t ELEM_SIZE = sizeof(scalar_t);
    static constexpr size_t ELEMS_PER_TILE = TILE;
};
using FloatTraits   = ScalarTraits<float>;
using IntTraits     = ScalarTraits<int>;
using ComplexTraits  = ScalarTraits<std::complex<float>>;


/**
 * This is a utility structure for maintaining an array aligned to ALIGNMENT boundaries in
 * memory.  This alignment should be at least TILE * ELEM_SIZE, though we make it even larger
 * here by default.
 */
template <typename scalar_t = float>
struct AlignedArray {
  AlignedArray(const size_t size) {
    int ret = posix_memalign((void**)&ptr, ALIGNMENT, size * sizeof(scalar_t));
    if (ret != 0) throw std::bad_alloc();
    this->size = size;
  }
  ~AlignedArray() { free(ptr); }
  size_t ptr_as_int() {return (size_t)ptr; }
  scalar_t* ptr;
  size_t size;
};



template <typename scalar_t>
void Fill(AlignedArray<scalar_t>* out, scalar_t val) {
  for (size_t i = 0; i < out->size; i++) {
    out->ptr[i] = val;
  }
}



template <typename scalar_t>
void Compact(const AlignedArray<scalar_t>& a, AlignedArray<scalar_t>* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory
   *
   * Args:
   *   a: non-compact representation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   *
   * Returns:
   *  void (you need to modify out directly, rather than returning anything; this is true for all the
   *  function will implement here, so we won't repeat this note.)
   */
  /// BEGIN SOLUTION
  int ndim = shape.size();
  size_t total = 1;
  for (int s : shape) total *= s;

  for (size_t i = 0; i < total; ++i) {
    size_t a_idx = offset;
    size_t temp = i;
    for (int d = ndim - 1; d >= 0; --d) {
      a_idx += (temp % shape[d]) * strides[d];
      temp /= shape[d];
    }
    out->ptr[i] = a.ptr[a_idx];
  }
  /// END SOLUTION
}

template <typename scalar_t>
void EwiseSetitem(const AlignedArray<scalar_t>& a, AlignedArray<scalar_t>* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array
   *
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN SOLUTION
  int ndim = shape.size();
  size_t total = 1;
  for (int s : shape) total *= s;

  for (size_t i = 0; i < total; ++i) {
    size_t out_idx = offset;
    size_t temp = i;
    for (int d = ndim - 1; d >= 0; --d) {
      out_idx += (temp % shape[d]) * strides[d];
      temp /= shape[d];
    }
    out->ptr[out_idx] = a.ptr[i];
  }
  /// END SOLUTION
}

template <typename scalar_t>
void ScalarSetitem(const size_t size, scalar_t val, AlignedArray<scalar_t>* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   *
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the
   *         product of items in shape, but convenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */

  /// BEGIN SOLUTION
  int ndim = shape.size();
  for (size_t i = 0; i < size; ++i) {
    size_t out_idx = offset;
    size_t temp = i;
    for (int d = ndim - 1; d >= 0; --d) {
      out_idx += (temp % shape[d]) * strides[d];
      temp /= shape[d];
    }
    out->ptr[out_idx] = val;
  }
  /// END SOLUTION
}

template <typename scalar_t>
void EwiseAdd(const AlignedArray<scalar_t>& a, const AlignedArray<scalar_t>& b, AlignedArray<scalar_t>* out) {
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + b.ptr[i];
  }
}

template <typename scalar_t>
void ScalarAdd(const AlignedArray<scalar_t>& a, scalar_t val, AlignedArray<scalar_t>* out) {
  /**
   * Set entries in out to be the sum of corresponding entry in a plus the scalar val.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + val;
  }
}


/**
 * In the code the follows, use the above template to create analogous element-wise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */
template <typename scalar_t, typename Op>
void EwiseOp(const AlignedArray<scalar_t>& a, const AlignedArray<scalar_t>& b,
             AlignedArray<scalar_t>* out, Op op) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = op(a.ptr[i], b.ptr[i]);
  }
}
template <typename T>
struct Maximum {
  T operator()(const T& a, const T& b) const { return std::max(a, b); }
};
template <typename scalar_t>
void EwiseMul(const AlignedArray<scalar_t>& a, const AlignedArray<scalar_t>& b, AlignedArray<scalar_t>* out) {
  EwiseOp(a, b, out, std::multiplies<scalar_t>{});
}

template <typename scalar_t>
void EwiseDiv(const AlignedArray<scalar_t>& a, const AlignedArray<scalar_t>& b, AlignedArray<scalar_t>* out) {
  EwiseOp(a, b, out, std::divides<scalar_t>{});
}

template <typename scalar_t>
void EwiseMaximum(const AlignedArray<scalar_t>& a, const AlignedArray<scalar_t>& b, AlignedArray<scalar_t>* out) {
  EwiseOp(a, b, out, Maximum<scalar_t>{});
}

template <typename scalar_t>
void EwiseEq(const AlignedArray<scalar_t>& a, const AlignedArray<scalar_t>& b, AlignedArray<scalar_t>* out) {
  EwiseOp(a, b, out, std::equal_to<scalar_t>{});
}

template <typename scalar_t>
void EwiseGe(const AlignedArray<scalar_t>& a, const AlignedArray<scalar_t>& b, AlignedArray<scalar_t>* out) {
  EwiseOp(a, b, out, std::greater_equal<scalar_t>{});
}


template <typename scalar_t, typename Op>
void EwiseUnaryOp(const AlignedArray<scalar_t>& a, AlignedArray<scalar_t>* out, Op op) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = op(a.ptr[i]);
  }
}
template <typename T>
struct Log {
  T operator()(const T& x) const { return std::log(x); }
};

template <typename T>
struct Exp {
  T operator()(const T& x) const { return std::exp(x); }
};

template <typename T>
struct Tanh {
  T operator()(const T& x) const { return std::tanh(x); }
};
template <typename scalar_t>
void EwiseLog(const AlignedArray<scalar_t>& a, AlignedArray<scalar_t>* out) {
  EwiseUnaryOp(a, out, Log<scalar_t>{});
}

template <typename scalar_t>
void EwiseExp(const AlignedArray<scalar_t>& a, AlignedArray<scalar_t>* out) {
  EwiseUnaryOp(a, out, Exp<scalar_t>{});
}

template <typename scalar_t>
void EwiseTanh(const AlignedArray<scalar_t>& a, AlignedArray<scalar_t>* out) {
  EwiseUnaryOp(a, out, Tanh<scalar_t>{});
}


template <typename scalar_t, typename Op>
void ScalarOp(const AlignedArray<scalar_t>& a, scalar_t val,
              AlignedArray<scalar_t>* out, Op op) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = op(a.ptr[i], val);
  }
}
template <typename T>
struct Power {
  T operator()(const T& base, const T& exp) const { return std::pow(base, exp); }
};
template <typename scalar_t>
void ScalarMul(const AlignedArray<scalar_t>& a, scalar_t val, AlignedArray<scalar_t>* out) {
  ScalarOp(a, val, out, std::multiplies<scalar_t>{});
}

template <typename scalar_t>
void ScalarDiv(const AlignedArray<scalar_t>& a, scalar_t val, AlignedArray<scalar_t>* out) {
  ScalarOp(a, val, out, std::divides<scalar_t>{});
}

template <typename scalar_t>
void ScalarPower(const AlignedArray<scalar_t>& a, scalar_t val, AlignedArray<scalar_t>* out) {
  ScalarOp(a, val, out, Power<scalar_t>{});
}

template <typename scalar_t>
void ScalarMaximum(const AlignedArray<scalar_t>& a, scalar_t val, AlignedArray<scalar_t>* out) {
  ScalarOp(a, val, out, Maximum<scalar_t>{});
}

template <typename scalar_t>
void ScalarEq(const AlignedArray<scalar_t>& a, scalar_t val, AlignedArray<scalar_t>* out) {
  ScalarOp(a, val, out, std::equal_to<scalar_t>{});
}

template <typename scalar_t>
void ScalarGe(const AlignedArray<scalar_t>& a, scalar_t val, AlignedArray<scalar_t>* out) {
  ScalarOp(a, val, out, std::greater_equal<scalar_t>{});
}

template <typename scalar_t>
void Matmul(const AlignedArray<scalar_t>& a, const AlignedArray<scalar_t>& b, AlignedArray<scalar_t>* out, uint32_t m, uint32_t n,
            uint32_t p) {
  /**
   * Multiply two (compact) matrices into an output (also compact) matrix.  For this implementation
   * you can use the "naive" three-loop algorithm.
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: compact 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   */

  /// BEGIN SOLUTION
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      out->ptr[i*p+j] = 0;
      for (size_t k = 0; k < n; k++) {
        out->ptr[i*p+j] += a.ptr[i*n+k]*b.ptr[k*p+j];
      }
    }
  }
  /// END SOLUTION
}

template <typename scalar_t>
inline void AlignedDot(const scalar_t* __restrict__ a,
                       const scalar_t* __restrict__ b,
                       scalar_t* __restrict__ out) {

  a = (const scalar_t*)__builtin_assume_aligned(a, TILE * sizeof(scalar_t));
  b = (const scalar_t*)__builtin_assume_aligned(b, TILE * sizeof(scalar_t));
  out = (scalar_t*)__builtin_assume_aligned(out, TILE * sizeof(scalar_t));

  for (size_t i = 0; i < TILE; i++) {
    for (size_t j = 0; j < TILE; j++) {
      for (size_t k = 0; k < TILE; k++) {
        out[i * TILE + j] += a[i * TILE + k] * b[k * TILE + j];
      }
    }
  }
}

template <typename scalar_t>
void MatmulTiled(const AlignedArray<scalar_t>& a, const AlignedArray<scalar_t>& b, AlignedArray<scalar_t>* out, uint32_t m,
                 uint32_t n, uint32_t p) {
  /**
   * Matrix multiplication on tiled representations of array.  In this setting, a, b, and out
   * are all *4D* compact arrays of the appropriate size, e.g. a is an array of size
   *   a[m/TILE][n/TILE][TILE][TILE]
   * You should do the multiplication tile-by-tile to improve performance of the array (i.e., this
   * function should call `AlignedDot()` implemented above).
   *
   * Note that this function will only be called when m, n, p are all multiples of TILE, so you can
   * assume that this division happens without any remainder.
   *
   * Args:
   *   a: compact 4D array of size m/TILE x n/TILE x TILE x TILE
   *   b: compact 4D array of size n/TILE x p/TILE x TILE x TILE
   *   out: compact 4D array of size m/TILE x p/TILE x TILE x TILE to write to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   *
   */
  /// BEGIN SOLUTION
  size_t TILE_SIZE = TILE * TILE;
  for (size_t idx = 0; idx < out->size; idx++) {
    out->ptr[idx] = 0;
  }
  for (size_t i = 0; i < m/TILE; i++) {
    for (size_t j = 0; j < p/TILE; j++) {
      size_t out_offset = (i*(p/TILE)+j)*TILE_SIZE;
      for (size_t k = 0; k < n/TILE; k++) {
        size_t a_offset = (i*(n/TILE)+k)*TILE_SIZE;
        size_t b_offset = (k*(p/TILE)+j)*TILE_SIZE;
        AlignedDot(a.ptr+a_offset, b.ptr+b_offset, out->ptr+out_offset);
      }
    }
  }
  /// END SOLUTION
}

template <typename scalar_t>
void ReduceMax(const AlignedArray<scalar_t>& a, AlignedArray<scalar_t>* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

  /// BEGIN SOLUTION
  for (size_t i = 0; i < out->size; i++) {
    out->ptr[i] = a.ptr[i*reduce_size];
    for (size_t j = i*reduce_size+1; j < (i+1)*reduce_size; j++) {
      out->ptr[i] = std::max(a.ptr[j], out->ptr[i]);
    }
  }
  /// END SOLUTION
}

template <typename scalar_t>
void ReduceSum(const AlignedArray<scalar_t>& a, AlignedArray<scalar_t>* out, size_t reduce_size) {
  /**
   * Reduce by taking sum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

  /// BEGIN SOLUTION
  for (size_t i = 0; i < out->size; i++) {
    out->ptr[i] = a.ptr[i*reduce_size];
    for (size_t j = i*reduce_size+1; j < (i+1)*reduce_size; j++) {
      out->ptr[i] = a.ptr[j] + out->ptr[i];
    }
  }
  /// END SOLUTION
}

}  // namespace cpu
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cpu, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cpu;

  m.attr("__device_name__") = "cpu";
  m.attr("__tile_size__") = TILE;

  py::class_<AlignedArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def("ptr", &AlignedArray::ptr_as_int)
      .def_readonly("size", &AlignedArray::size);

  // return numpy array (with copying for simplicity, otherwise garbage
  // collection is a pain)
  m.def("to_numpy", [](const AlignedArray& a, std::vector<size_t> shape,
                       std::vector<size_t> strides, size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });
    return py::array_t<scalar_t>(shape, numpy_strides, a.ptr + offset);
  });

  // convert from numpy (with copying)
  m.def("from_numpy", [](py::array_t<scalar_t> a, AlignedArray* out) {
    std::memcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE);
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);
  m.def("matmul_tiled", MatmulTiled);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
