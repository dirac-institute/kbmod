#ifndef PSI_PHI_ARRAY_DOCS
#define PSI_PHI_ARRAY_DOCS

namespace pydocs {

static const auto DOC_PsiPhi = R"doc(
  A named tuple for psi and phi values.

  Attributes
  ----------
  psi : `float`
      The psi value at a pixel.
  phi : `float`
      The phi value at a pixel.
  )doc";

static const auto DOC_PsiPhiArray = R"doc(
  An encoded array of Psi and Phi values along with their meta data.
  )doc";

static const auto DOC_PsiPhiArray_get_num_bytes = R"doc(
  The target number of bytes to use for encoding the data (1 for uint8, 2 for uint16,
  or 4 for float32). Might differ from actual number of bytes (block_size).
  )doc";

static const auto DOC_PsiPhiArray_get_num_times = R"doc(
  The number of times.
  )doc";

static const auto DOC_PsiPhiArray_get_width = R"doc(
  The image width.
  )doc";

static const auto DOC_PsiPhiArray_get_height = R"doc(
  The image height.
  )doc";

static const auto DOC_PsiPhiArray_get_pixels_per_image = R"doc(
  The number of pixels per each image.
  )doc";

static const auto DOC_PsiPhiArray_get_num_entries = R"doc(
  The number of array entries.
  )doc";

static const auto DOC_PsiPhiArray_get_total_array_size = R"doc(
  The size of the array in bytes.
  )doc";

static const auto DOC_PsiPhiArray_get_block_size = R"doc(
  The size of a single entry in bytes.
  )doc";

static const auto DOC_PsiPhiArray_get_psi_min_val = R"doc(
  The minimum value of psi used in the scaling computations.
  )doc";

static const auto DOC_PsiPhiArray_get_psi_max_val = R"doc(
  The maximum value of psi used in the scaling computations.
  )doc";

static const auto DOC_PsiPhiArray_get_psi_scale = R"doc(
  The scaling parameter for psi.
  )doc";

static const auto DOC_PsiPhiArray_get_phi_min_val = R"doc(
  The minimum value of phi used in the scaling computations.
  )doc";

static const auto DOC_PsiPhiArray_get_phi_max_val = R"doc(
  The maximum value of phi used in the scaling computations.
  )doc";

static const auto DOC_PsiPhiArray_get_phi_scale = R"doc(
  The scaling parameter for phi.
  )doc";

static const auto DOC_PsiPhiArray_get_cpu_array_allocated = R"doc(
  A Boolean indicating whether the cpu array exists.
  )doc";

static const auto DOC_PsiPhiArray_get_gpu_array_allocated = R"doc(
  A Boolean indicating whether the gpu array exists.
  )doc";

static const auto DOC_PsiPhiArray_clear = R"doc(
  Clear all data and free the arrays.
  )doc";

static const auto DOC_PsiPhiArray_read_psi_phi = R"doc(
  Read a PsiPhi value from the CPU array.

  Parameters
  ----------
  time : `int`
      The timestep to read.
  row : `int`
      The row in the image (y-dimension) 
  col : `int`
      The column in the image (x-dimension) 

  Returns
  -------
  `PsiPhi`
      The pixel values.
  )doc";

static const auto DOC_PsiPhiArray_set_meta_data = R"doc(
    Set the meta data for the array. Automatically called by
    fill_psi_phi_array().

    Parameters
    ----------
    new_num_bytes : `int`
        The type of encoding to use (1, 2, or 4).
    new_num_times : `int`
        The number of time steps in the data.
    new_height : `int`
        The height of each image in pixels.
    new_width : `int`
        The width of each image in pixels.
  )doc";

static const auto DOC_PsiPhiArray_fill_psi_phi_array = R"doc(
    Fill the PsiPhiArray from Psi and Phi images.

    Parameters
    ----------
    result_data : `PsiPhiArray`
        The location to store the data.
    num_bytes : `int`
        The type of encoding to use (1, 2, or 4).
    psi_imgs : `list`
        A list of psi images.
    phi_imgs : `list`
        A list of phi images.
  )doc";

}  // namespace pydocs

#endif /* PSI_PHI_ARRAY_DOCS */
