#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

// Forward declarations for cache functions (defined in tensordot_bs.cu)
namespace cublocksparse {
    void clear_plan_cache();
    size_t plan_cache_size();
    std::vector<std::string> plan_cache_keys();
    std::vector<std::pair<std::string, size_t>> plan_cache_stats();
}

static PyObject* py_clear_plan_cache(PyObject* self, PyObject* args) {
    cublocksparse::clear_plan_cache();
    Py_RETURN_NONE;
}

static PyObject* py_plan_cache_size(PyObject* self, PyObject* args) {
    return PyLong_FromSize_t(cublocksparse::plan_cache_size());
}

static PyObject* py_plan_cache_keys(PyObject* self, PyObject* args) {
    auto keys = cublocksparse::plan_cache_keys();
    PyObject* list = PyList_New(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        PyList_SetItem(list, i, PyUnicode_FromString(keys[i].c_str()));
    }
    return list;
}

static PyObject* py_plan_cache_stats(PyObject* self, PyObject* args) {
    auto stats = cublocksparse::plan_cache_stats();
    PyObject* dict = PyDict_New();
    for (const auto& [key, hits] : stats) {
        PyDict_SetItemString(dict, key.c_str(), PyLong_FromSize_t(hits));
    }
    return dict;
}

static PyMethodDef module_methods[] = {
    {"clear_plan_cache", py_clear_plan_cache, METH_NOARGS, 
     "Clear the cuTENSOR plan cache"},
    {"plan_cache_size", py_plan_cache_size, METH_NOARGS, 
     "Get number of cached plans"},
    {"plan_cache_keys", py_plan_cache_keys, METH_NOARGS, 
     "Get list of cache keys"},
    {"plan_cache_stats", py_plan_cache_stats, METH_NOARGS, 
     "Get dict mapping cache keys to hit counts"},
    {NULL, NULL, 0, NULL}
};

extern "C" {
  /* Creates a dummy empty _C module that can be imported from Python.
    The import from Python will load the .so consisting of this file
    in this extension, so that the TORCH_LIBRARY static initializers
    below are run. */
  PyObject* PyInit__C(void)
  {
      static struct PyModuleDef module_def = {
          PyModuleDef_HEAD_INIT,
          "_C",   /* name of module */
          NULL,   /* module documentation, may be NULL */
          -1,     /* size of per-interpreter state of the module,
                    or -1 if the module keeps state in global variables. */
          module_methods,   /* methods */
      };
      return PyModule_Create(&module_def);
  }
}

namespace cublocksparse {

TORCH_LIBRARY(cublocksparse, m) {
  // See https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit?tab=t.0

  // int[int[]] int[][] not supported in PyTorch Custom OPS API, we use Tensor instead
  // m.def("tensordot_bs(Tensor a, Tensor b, "
  //     "int[int[]] a_blocks, int[int[]] a_D_per_mode, int[] nout_a, int[] nin_a, "
  //     "int[int[]] b_blocks, int[int[]] b_D_per_mode, int[] nout_b, int[] nin_b, "
  //     "int c_size, int[int[]] c_blocks, int[int[]] c_D_per_mode) -> Tensor"
  // );

  m.def("tensordot_bs(Tensor a, Tensor b, "
    "int[] a_blocks, int[] a_offsets, int[] a_strides, Tensor a_D_per_mode, int[] nout_a, int[] nin_a, "
    "int[] b_blocks, int[] b_offsets, int[] b_strides, Tensor b_D_per_mode, int[] nout_b, int[] nin_b, "
    "int c_size, int[] c_blocks, int[] c_offsets, int[] c_strides, Tensor c_D_per_mode) -> Tensor"
  );
}

}