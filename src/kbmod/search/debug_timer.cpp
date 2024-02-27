/* A very simple class that wraps debug output with timing. Used for
   providing status updates as the system is running if in verbose
   mode and nothing otherwise.

   This is *not* a high precision timer meant to be used for benchmarking.
*/

#include "debug_timer.h"
#include "pydocs/debug_timer_docs.h"

namespace search {

DebugTimer::DebugTimer(std::string message, std::string name)
        : message_(message), logger_(logging::getLogger(name)) {
    start();
}

DebugTimer::DebugTimer(std::string message, logging::Logger* logger) : message_(message), logger_(logger) {
    start();
}

DebugTimer::DebugTimer(std::string message) : message_(message) {
    std::replace(message.begin(), message.end(), ' ', '.');
    std::string derived_name = "DebugTimer." + message;
    logger_ = logging::getLogger(derived_name);
    start();
}

void DebugTimer::start() {
    running_ = true;
    t_start_ = std::chrono::system_clock::now();
    logger_->debug("Starting " + message_ + " timer.");
}

void DebugTimer::stop() {
    t_end_ = std::chrono::system_clock::now();
    running_ = false;
    auto t_delta = std::chrono::duration_cast<std::chrono::milliseconds>(t_end_ - t_start_);
    logger_->debug("Finished " + message_ + " in " + std::to_string(t_delta.count() / 1000.0) + "seconds.");
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
    logger_->debug("Step " + message_ + " is at " + std::to_string(result) + "seconds.");
    return result;
}

#ifdef Py_PYTHON_H
static void debug_timer_binding(py::module& m) {
    using dbt = search::DebugTimer;
    py::class_<dbt>(m, "DebugTimer", pydocs::DOC_DEBUG_TIMER)
            .def(py::init<std::string, std::string>())
            .def(py::init<std::string>())
            .def(py::init([](std::string message, py::object logger) {
                std::string name = std::string(py::str(logger.attr("name")));
                return std::unique_ptr<DebugTimer>(new DebugTimer(message, name));
            }))
            .def("start", &dbt::start, pydocs::DOC_DEBUG_TIMER_start)
            .def("stop", &dbt::stop, pydocs::DOC_DEBUG_TIMER_stop)
            .def("read", &dbt::read, pydocs::DOC_DEBUG_TIMER_read);
}
#endif /* Py_PYTHON_H */

} /* namespace search */
