#ifndef GPU_DOCS
#define GPU_DOCS

namespace pydocs {

static const auto DOC_print_cuda_stats = R"doc(
  Display the basic GPU information.
  )doc";

static const auto DOC_validate_gpu = R"doc(
  Check that a GPU is present, accessible, and has sufficient memory.

  Parameters
  ----------
  req_memory : `int`
      The minimum free memory in bytes.
      Default: 0

  Returns
  -------
  `bool`
     Indicates whether the GPU is valid.
  )doc";

}  // namespace pydocs

#endif /* GPU_DOCS */
