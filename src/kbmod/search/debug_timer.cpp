/* A very simple class that wraps debug output with timing. Used for
   providing status updates as the system is running if in verbose
   mode and nothing otherwise.

   This is *not* a high precision timer meant to be used for benchmarking.
*/

#include "debug_timer.h"
#include "pydocs/debug_timer_docs.h"

namespace search {

DebugTimer::DebugTimer(std::string name, bool verbose) {
    name_ = name;
    verbose_ = verbose;
    start();
}

void DebugTimer::start() {
    running_ = true;
    t_start_ = std::chrono::system_clock::now();
    if (verbose_) {
        std::cout << "Starting " << name_ << "...\n" << std::flush;
    }
}

void DebugTimer::stop() {
    t_end_ = std::chrono::system_clock::now();
    running_ = false;

    if (verbose_) {
        auto t_delta = std::chrono::duration_cast<std::chrono::milliseconds>(t_end_ - t_start_);
        std::cout << name_ << " finished in " << t_delta.count() / 1000.0 << " seconds.\n" << std::flush;
    }
}

double DebugTimer::read() {
    std::chrono::milliseconds t_delta;
    if (running_) {
        std::chrono::time_point<std::chrono::system_clock> t_current_ = std::chrono::system_clock::now();
        t_delta = std::chrono::duration_cast<std::chrono::milliseconds>(t_current_ - t_start_);
    } else {
        t_delta = std::chrono::duration_cast<std::chrono::milliseconds>(t_end_ - t_start_);
    }

    double result = t_delta.count() / 1000.0;
    if (verbose_) {
        std::cout << name_ << " at " << result << " seconds.\n" << std::flush;
    }
    return result;
}

#ifdef Py_PYTHON_H
static void debug_timer_binding(py::module& m) {
    using dbt = search::DebugTimer;
    py::class_<dbt>(m, "DebugTimer", pydocs::DOC_DEBUG_TIMER)
            .def(py::init<std::string, bool>())
            .def("start", &dbt::start, pydocs::DOC_DEBUG_TIMER_start)
            .def("stop", &dbt::stop, pydocs::DOC_DEBUG_TIMER_stop)
            .def("read", &dbt::read, pydocs::DOC_DEBUG_TIMER_read);
}
#endif /* Py_PYTHON_H */

} /* namespace search */
