#ifndef STAMPCREATOR_H_
#define STAMPCREATOR_H_

#include "common.h"
#include "image_stack.h"
#include "pydocs/stamp_creator_docs.h"

namespace search {
class StampCreator {
public:
    StampCreator();

    // Functions for creating science stamps for filtering, visualization, etc. User can specify
    // the radius of the stamp, whether to interpolate among pixels, whether to keep NO_DATA values
    // or replace them with zero, and what indices to use.
    // The indices to use are indicated by use_index: a vector<bool> indicating whether to use
    // each time step. An empty (size=0) vector will use all time steps.
    std::vector<RawImage> create_stamps(ImageStack stack, const Trajectory& trj, int radius, bool interpolate,
                                        bool keep_no_data, const std::vector<bool>& use_index);

    
    std::vector<RawImage> create_stamps(ImageStack stack, Trajectory t, int radius, const std::vector<RawImage*>& imgs,
                                                   bool interpolate);

    virtual ~StampCreator(){};
};

} /* namespace search */

#endif /* STAMPCREATOR_H_ */
