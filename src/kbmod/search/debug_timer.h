/* A very simple class that wraps debug output with timing. Used for
   providing status updates as the system is running if in verbose
   mode and nothing otherwise.
*/
#ifndef DEBUG_TIMER_H_
#define DEBUG_TIMER_H_

#include <iostream>
#include <chrono>

namespace search {
class DebugTimer {
public:
    DebugTimer(std::string name, bool verbose) {
        name_ = name;
        verbose_ = verbose;
        start();
    }

    void start() {
        t_start_ = std::chrono::system_clock::now();
        if (verbose_) {
            std::cout << "Starting " << name_ << "...\n" << std::flush;
        }
    }

    void stop() {
        if (verbose_) {
            std::chrono::time_point<std::chrono::system_clock> t_end = std::chrono::system_clock::now();
            auto t_delta = std::chrono::duration_cast<std::chrono::seconds>(t_end - t_start_);
            std::cout << name_ << " took " << t_delta.count() << " seconds." << std::flush;
        }
    }

private:
    std::chrono::time_point<std::chrono::system_clock> t_start_;
    std::string name_;
    bool verbose_;
};

} /* namespace search */

#endif /* DEBUG_TIMER_H_ */
