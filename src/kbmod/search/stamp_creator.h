#ifndef STAMPCREATOR_H_
#define STAMPCREATOR_H_

#include "common.h"
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
    // the radius of the stamp, whether to keep NO_DATA values or replace them with zero,
    // and what indices to use.
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

    virtual ~StampCreator(){};
};

} /* namespace search */

#endif /* STAMPCREATOR_H_ */
