#ifndef GPU_DOCS
#define GPU_DOCS

namespace pydocs {

static const auto DOC_print_cuda_stats = R"doc(
  Display the basic GPU information to standard out.
  )doc";

static const auto DOC_get_gpu_free_memory = R"doc(
  Return the GPUs free memory in bytes.
  )doc";

static const auto DOC_get_gpu_total_memory = R"doc(
  Return the GPUs total memory in bytes.
  )doc";

static const auto DOC_stat_gpu_memory_mb = R"doc(
  Create a minimal GPU stats string for debugging.
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
