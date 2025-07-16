#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

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
          NULL,   /* methods */
      };
      return PyModule_Create(&module_def);
  }
}

namespace cublocksparse {

// TORCH_LIBRARY(cublocksparse, m) {
//    // Note that "float" in the schema corresponds to the C++ double type
//    // and the Python float type.
//    m.def("tensordot_bs(Tensor a, Tensor b) -> Tensor");
// }

TORCH_LIBRARY(cublocksparse, m) {
   // See https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit?tab=t.0

   // int[int[]] int[][] not supported in PyTorch Custom OPS API
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
//   m.def("tensordot_bs(Tensor a, Tensor b, "
//     "int[] a_blocks, int[] a_D_per_mode, int[] nout_a, int[] nin_a, "
//     "int[] b_blocks, int[] b_D_per_mode, int[] nout_b, int[] nin_b, "
//     "int c_size, int[] c_blocks, int[] c_D_per_mode) -> Tensor"
//   );
}

}