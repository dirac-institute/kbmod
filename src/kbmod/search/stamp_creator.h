#ifndef STAMPCREATOR_H_
#define STAMPCREATOR_H_

#include "common.h"
#include "gpu_array.h"
#include "image_stack.h"
#include "pydocs/stamp_creator_docs.h"

namespace search {
/**
 * Utility class for functions used for creating science stamps for
 * filtering, visualization, etc.
 */
class StampCreator {
public:
    StampCreator();

    // Functions science stamps for filtering, visualization, etc. User can specify
    // the radius of the stamp, whether to keep no data values (e.g. NaN) or replace
    //  them with zero, and what indices to use.
    // The indices to use are indicated by use_index: a vector<bool> indicating whether to use
    // each time step. An empty (size=0) vector will use all time steps.
    static std::vector<RawImage> create_stamps(ImageStack& stack, const Trajectory& trj, int radius,
                                               bool keep_no_data, const std::vector<bool>& use_index);

    static std::vector<RawImage> get_stamps(ImageStack& stack, const Trajectory& t, int radius);

    static RawImage get_median_stamp(ImageStack& stack, const Trajectory& trj, int radius,
                                     const std::vector<bool>& use_index);

    static RawImage get_mean_stamp(ImageStack& stack, const Trajectory& trj, int radius,
                                   const std::vector<bool>& use_index);

    static RawImage get_summed_stamp(ImageStack& stack, const Trajectory& trj, int radius,
                                     const std::vector<bool>& use_index);

    // Compute a mean or summed stamp for each trajectory on the GPU or CPU.
    // The GPU implementation is slower for small numbers of trajectories (< 500), but performs
    // relatively better as the number of trajectories increases. If filtering is applied then
    // the code will return a 1x1 image with NO_DATA to represent each filtered image.
    static std::vector<RawImage> get_coadded_stamps(ImageStack& stack, std::vector<Trajectory>& t_array,
                                                    std::vector<std::vector<bool> >& use_index_vect,
                                                    const StampParameters& params, bool use_gpu);

    static std::vector<RawImage> get_coadded_stamps_gpu(ImageStack& stack, std::vector<Trajectory>& t_array,
                                                        std::vector<std::vector<bool> >& use_index_vect,
                                                        const StampParameters& params);

    static std::vector<RawImage> get_coadded_stamps_cpu(ImageStack& stack, std::vector<Trajectory>& t_array,
                                                        std::vector<std::vector<bool> >& use_index_vect,
                                                        const StampParameters& params);

    // Function to do the actual stamp filtering.
    static bool filter_stamp(const RawImage& img, const StampParameters& params);

    // Functions for generating variance stamps. All times are returns and NO_DATA values are preserved.
    static std::vector<RawImage> create_variance_stamps(ImageStack& stack, const Trajectory& trj, int radius);

    virtual ~StampCreator(){};
};

} /* namespace search */

#endif /* STAMPCREATOR_H_ */
