/* A very simple class that wraps debug output with timing. Used for
   providing status updates as the system is running if in verbose
   mode and nothing otherwise.

   This is *not* a high precision timer meant to be used for benchmarking.
*/
#ifndef DEBUG_TIMER_H_
#define DEBUG_TIMER_H_

#include <iostream>
#include <chrono>

#include "logging.h"


namespace search {
class DebugTimer {
public:
    DebugTimer(std::string name, bool verbose);

    void start();
    void stop();

    // Read the time in decimal seconds without stopping the timer.
    // If the timer is already stopped, read the duration from end time.
    double read();

  private:
    std::chrono::time_point<std::chrono::system_clock> t_start_;
    std::chrono::time_point<std::chrono::system_clock> t_end_;
    std::string name_;
    bool verbose_;
    bool running_;
};

} /* namespace search */

#endif /* DEBUG_TIMER_H_ */
