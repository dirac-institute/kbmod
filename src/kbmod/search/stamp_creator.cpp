#include "stamp_creator.h"

namespace search {
#ifdef HAVE_CUDA
void deviceGetCoadds(const uint64_t num_images, const uint64_t width, const uint64_t height,
                     GPUArray<float>& image_data, GPUArray<double>& image_times,
                     GPUArray<Trajectory>& trajectories, StampParameters params,
                     GPUArray<int>& use_index_vect, GPUArray<float>& results);
#endif

std::vector<RawImage> create_stamps(ImageStack& stack, const Trajectory& trj, int radius, bool keep_no_data,
                                    const std::vector<bool>& use_index) {
    if (use_index.size() > 0)
        assert_sizes_equal(use_index.size(), stack.img_count(), "create_stamps() use_index");
    bool use_all_stamps = (use_index.size() == 0);

    std::vector<RawImage> stamps;
    unsigned int num_times = stack.img_count();
    for (unsigned int i = 0; i < num_times; ++i) {
        if (use_all_stamps || use_index[i]) {
            // Calculate the trajectory position.
            double time = stack.get_zeroed_time(i);
            Point pos{trj.get_x_pos(time), trj.get_y_pos(time)};
            RawImage& img = stack.get_single_image(i).get_science();

            RawImage stamp = img.create_stamp(pos, radius, keep_no_data);
            stamps.push_back(std::move(stamp));
        }
    }
    return stamps;
}

std::vector<RawImage> create_stamps_xy(ImageStack& stack, int radius, const std::vector<int>& xvals,
                                       const std::vector<int>& yvals, const std::vector<int>& image_indices) {
    // Check if we are using all indices or just the ones in image_indices.
    unsigned int num_stamps = stack.img_count();
    bool use_inds = false;
    if (image_indices.size() > 0) {
        num_stamps = image_indices.size();
        use_inds = true;
    }

    // Make sure that the x and y values are the same size as the number of
    // stamps to generate.
    assert_sizes_equal(xvals.size(), num_stamps, "xvals");
    assert_sizes_equal(yvals.size(), num_stamps, "yvals");

    std::vector<RawImage> stamps;
    for (unsigned int i = 0; i < num_stamps; ++i) {
        unsigned int ind = (use_inds) ? image_indices[i] : i;

        Point pos{xvals[i], yvals[i]};
        RawImage& img = stack.get_single_image(ind).get_science();

        RawImage stamp = img.create_stamp(pos, radius, false);
        stamps.push_back(std::move(stamp));
    }
    return stamps;
}

// For stamps used for visualization we replace invalid pixels with zeros
// and return all the stamps (regardless of whether individual timesteps
// have been filtered).
std::vector<RawImage> get_stamps(ImageStack& stack, const Trajectory& t, int radius) {
    std::vector<bool> empty_vect;
    return create_stamps(stack, t, radius, false /*=keep_no_data*/, empty_vect);
}

// For creating coadded stamps, we keep invalid pixels tagged (so we can filter it out of mean/median).
RawImage get_median_stamp(ImageStack& stack, const Trajectory& trj, int radius,
                          const std::vector<bool>& use_index) {
    return create_median_image(create_stamps(stack, trj, radius, true /*=keep_no_data*/, use_index));
}

// For creating coadded stamps, we keep invalid pixels tagged (so we can filter it out of mean/median).
RawImage get_mean_stamp(ImageStack& stack, const Trajectory& trj, int radius,
                        const std::vector<bool>& use_index) {
    return create_mean_image(create_stamps(stack, trj, radius, true /*=keep_no_data*/, use_index));
}

// For creating summed stamps, we replace invalid pixels with zero (which is the same as
// filtering it out for the sum).
RawImage get_summed_stamp(ImageStack& stack, const Trajectory& trj, int radius,
                          const std::vector<bool>& use_index) {
    return create_summed_image(create_stamps(stack, trj, radius, false /*=keep_no_data*/, use_index));
}

std::vector<RawImage> get_coadded_stamps(ImageStack& stack, std::vector<Trajectory>& t_array,
                                         std::vector<std::vector<bool>>& use_index_vect,
                                         const StampParameters& params, bool use_gpu) {
    logging::Logger* rs_logger = logging::getLogger("kbmod.search.stamp_creator");
    rs_logger->info("Generating co_added stamps on " + std::to_string(t_array.size()) + " trajectories.");
    DebugTimer timer = DebugTimer("coadd generating", rs_logger);

    // If the stamps are larger than the GPU can handle, fall back to CPU.
    if (use_gpu && params.radius >= 15) {
        rs_logger->info("Stamp size too large for GPU. Performing co-adds on the CPU.");
        use_gpu = false;
    }

    // We currently can only generate the image stamps (not the variance) on GPU.
    if (use_gpu && (params.stamp_type == STAMP_VAR_WEIGHTED)) {
        rs_logger->info("Performing variance weighted co-adds on the CPU.");
        use_gpu = false;
    }

    // We use the GPU if we have it for everything except STAMP_VAR_WEIGHTED which is CPU only.
    if (use_gpu) {
#ifdef HAVE_CUDA
        rs_logger->info("Performing co-adds on the GPU.");
        return get_coadded_stamps_gpu(stack, t_array, use_index_vect, params);
#else
        rs_logger->warning("GPU is not enabled. Performing co-adds on the CPU.");
#endif
    }
    return get_coadded_stamps_cpu(stack, t_array, use_index_vect, params);
}

std::vector<RawImage> get_coadded_stamps_cpu(ImageStack& stack, std::vector<Trajectory>& t_array,
                                             std::vector<std::vector<bool>>& use_index_vect,
                                             const StampParameters& params) {
    const uint64_t num_trajectories = t_array.size();
    std::vector<RawImage> results(num_trajectories);

    for (uint64_t i = 0; i < num_trajectories; ++i) {
        RawImage coadd(1, 1);
        switch (params.stamp_type) {
            case STAMP_MEDIAN:
                coadd = get_median_stamp(stack, t_array[i], params.radius, use_index_vect[i]);
                break;
            case STAMP_MEAN:
                coadd = get_mean_stamp(stack, t_array[i], params.radius, use_index_vect[i]);
                break;
            case STAMP_SUM:
                coadd = get_summed_stamp(stack, t_array[i], params.radius, use_index_vect[i]);
                break;
            case STAMP_VAR_WEIGHTED:
                coadd = get_variance_weighted_stamp(stack, t_array[i], params.radius, use_index_vect[i]);
                break;
            default:
                throw std::runtime_error("Invalid stamp coadd type.");
        }
        results[i] = std::move(coadd);
    }

    return results;
}

std::vector<RawImage> get_coadded_stamps_gpu(ImageStack& stack, std::vector<Trajectory>& t_array,
                                             std::vector<std::vector<bool>>& use_index_vect,
                                             const StampParameters& params) {
    logging::Logger* rs_logger = logging::getLogger("kbmod.search.stamp_creator");

    // Right now only limited stamp sizes are allowed.
    if (2 * params.radius + 1 > MAX_STAMP_EDGE || params.radius <= 0) {
        throw std::runtime_error("Invalid stamp radius=" + std::to_string(params.radius));
    }

    const unsigned int num_images = stack.img_count();
    const unsigned int width = stack.get_width();
    const unsigned int height = stack.get_height();

    // Allocate space for the results.
    const uint64_t num_trajectories = t_array.size();
    const uint64_t stamp_width = 2 * params.radius + 1;
    const uint64_t stamp_ppi = stamp_width * stamp_width;
    const uint64_t total_stamp_pixels = stamp_ppi * num_trajectories;
    const uint64_t stamp_bytes = total_stamp_pixels * sizeof(float);
    rs_logger->debug("Allocating CPU memory for " + std::to_string(num_trajectories) + " stamps with " +
                     std::to_string(total_stamp_pixels) + " pixels. Using " + std::to_string(stamp_bytes) +
                     " bytes");
    std::vector<float> stamp_data(total_stamp_pixels);

    // Do the co-adds.
#ifdef HAVE_CUDA
    bool was_on_gpu = stack.on_gpu();
    if (!was_on_gpu) {
        rs_logger->debug("Moving images onto GPU.");
        stack.copy_to_gpu();
    }

    // Create the on other on-GPU data structures. We do that here (instead of in the CUDA)
    // code so we can log debugging information.
    GPUArray<float> device_stamps(total_stamp_pixels);
    rs_logger->debug("Allocating GPU memory for stamps. " + device_stamps.stats_string());
    device_stamps.allocate_gpu_memory();

    GPUArray<Trajectory> device_trjs(num_trajectories);
    rs_logger->debug("Allocating GPU and copying memory for trajectories. " + device_trjs.stats_string());
    device_trjs.copy_vector_to_gpu(t_array);

    // Check if we need to create a vector of per-trajectory, per-image use.
    // Convert the vector of booleans into an integer array so we do a cudaMemcpy.
    GPUArray<int> device_use_index(num_trajectories * num_images);
    if (use_index_vect.size() == num_trajectories) {
        rs_logger->debug("Allocating GPU memory for vector of indices to use. " +
                         device_use_index.stats_string());
        device_use_index.allocate_gpu_memory();

        // Copy the data into the GPU in chunks so we don't have to allocate the
        // space for all of the integer arrays on the CPU side as well.
        std::vector<int> int_vect(num_images, 0);
        for (uint64_t i = 0; i < num_trajectories; ++i) {
            if (use_index_vect[i].size() != num_images) {
                throw std::runtime_error("Number of images and indices do not match");
            }
            for (unsigned t = 0; t < num_images; ++t) {
                int_vect[t] = use_index_vect[i][t] ? 1 : 0;
            }
            device_use_index.copy_vector_into_subset_of_gpu(int_vect, i * num_images);
        }
    } else {
        rs_logger->debug("Not using 'use_index_vect'");
    }

    deviceGetCoadds(num_images, width, height, stack.get_gpu_image_array(), stack.get_gpu_time_array(),
                    device_trjs, params, device_use_index, device_stamps);

    // Read back results from the GPU.
    rs_logger->debug("Moving stamps to CPU.");
    device_stamps.copy_gpu_to_vector(stamp_data);

    // Clean up the memory. If we put the data on GPU this function, make sure to clean it up.
    rs_logger->debug("Freeing GPU stamp memory. " + device_stamps.stats_string());
    device_stamps.free_gpu_memory();
    rs_logger->debug("Freeing GPU trajectory memory. " + device_trjs.stats_string());
    device_trjs.free_gpu_memory();
    if (device_use_index.on_gpu()) {
        rs_logger->debug("Freeing GPU 'use_index_vect' memory. " + device_use_index.stats_string());
        device_use_index.free_gpu_memory();
    }
    if (!was_on_gpu) {
        rs_logger->debug("Freeing GPU image memory.");
        stack.clear_from_gpu();
    }
#else
    throw std::runtime_error("Non-GPU co-adds is not implemented.");
#endif

    // Copy the stamps into RawImages.
    std::vector<RawImage> results(num_trajectories);
    std::vector<float> current_pixels(stamp_ppi, 0.0);
    for (uint64_t t = 0; t < num_trajectories; ++t) {
        // Copy the data into a single RawImage.
        uint64_t offset = t * stamp_ppi;
        for (uint64_t p = 0; p < stamp_ppi; ++p) {
            current_pixels[p] = stamp_data[offset + p];
        }

        Image tmp = Eigen::Map<Image>(current_pixels.data(), stamp_width, stamp_width);
        RawImage current_image = RawImage(tmp);
        results[t] = std::move(current_image);
    }
    return results;
}

std::vector<RawImage> create_variance_stamps(ImageStack& stack, const Trajectory& trj, int radius,
                                             const std::vector<bool>& use_index) {
    if (use_index.size() > 0)
        assert_sizes_equal(use_index.size(), stack.img_count(), "create_stamps() use_index");
    bool use_all_stamps = (use_index.size() == 0);

    std::vector<RawImage> stamps;
    unsigned int num_times = stack.img_count();
    for (unsigned int i = 0; i < num_times; ++i) {
        if (use_all_stamps || use_index[i]) {
            // Calculate the trajectory position.
            double time = stack.get_zeroed_time(i);
            Point pos{trj.get_x_pos(time), trj.get_y_pos(time)};
            RawImage& img = stack.get_single_image(i).get_variance();

            RawImage stamp = img.create_stamp(pos, radius, true /* keep_no_data */);
            stamps.push_back(std::move(stamp));
        }
    }
    return stamps;
}

RawImage get_variance_weighted_stamp(ImageStack& stack, const Trajectory& trj, int radius,
                                     const std::vector<bool>& use_index) {
    if (radius < 0) throw std::runtime_error("Invalid stamp radius. Must be >= 0.");
    unsigned int num_images = stack.img_count();
    if (num_images == 0) throw std::runtime_error("Unable to create mean image given 0 images.");
    unsigned int stamp_width = 2 * radius + 1;

    // Make the stamps for each time step.
    std::vector<RawImage> sci_stamps = create_stamps(stack, trj, radius, true /*=keep_no_data*/, use_index);
    std::vector<RawImage> var_stamps = create_variance_stamps(stack, trj, radius, use_index);
    if (sci_stamps.size() != var_stamps.size()) {
        throw std::runtime_error("Mismatched number of stamps returned.");
    }
    num_images = sci_stamps.size();

    // Do the weighted mean.
    Image result = Image::Zero(stamp_width, stamp_width);
    for (unsigned int y = 0; y < stamp_width; ++y) {
        for (unsigned int x = 0; x < stamp_width; ++x) {
            float sum = 0.0;
            float scale = 0.0;
            for (unsigned int i = 0; i < num_images; ++i) {
                float sci_val = sci_stamps[i].get_pixel({y, x});
                float var_val = var_stamps[i].get_pixel({y, x});
                if (pixel_value_valid(sci_val) && pixel_value_valid(var_val) && (var_val != 0.0)) {
                    sum += sci_val / var_val;
                    scale += 1.0 / var_val;
                }
            }

            if (scale > 0.0) {
                result(y, x) = sum / scale;
            } else {
                result(y, x) = 0.0;
            }
        }  // for x
    }      // for y
    return RawImage(result);
}

#ifdef Py_PYTHON_H
static void stamp_creator_bindings(py::module& m) {
    m.def("get_stamps", &search::get_stamps, pydocs::DOC_StampCreator_get_stamps);
    m.def("get_median_stamp", &search::get_median_stamp, pydocs::DOC_StampCreator_get_median_stamp);
    m.def("get_mean_stamp", &search::get_mean_stamp, pydocs::DOC_StampCreator_get_mean_stamp);
    m.def("get_summed_stamp", &search::get_summed_stamp, pydocs::DOC_StampCreator_get_summed_stamp);
    m.def("get_coadded_stamps", &search::get_coadded_stamps, pydocs::DOC_StampCreator_get_coadded_stamps);
    m.def("get_variance_weighted_stamp", &search::get_variance_weighted_stamp,
          pydocs::DOC_StampCreator_get_variance_weighted_stamp);
    m.def("create_stamps", &search::create_stamps, pydocs::DOC_StampCreator_create_stamps);
    m.def("create_stamps_xy", &search::create_stamps_xy, pydocs::DOC_StampCreator_create_stamps_xy);
    m.def("create_variance_stamps", &search::create_variance_stamps,
          pydocs::DOC_StampCreator_create_variance_stamps);
}
#endif /* Py_PYTHON_H */

} /* namespace search */
