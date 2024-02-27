#ifndef KBMOD_LOGGER
#define KBMOD_LOGGER

#include <iostream>
#include <iomanip>
#include <sstream>
#include <regex>
#include <unordered_map>
#include <string>
#include <vector>

/*
 * The Logging class is a singleton that keeps a reference to all created
 * Loggers. The Loggers define the log format and IO method (stdout, file etc.).
 * Logging keeps references to Loggers in a registry. This registry is exposed
 * via the `getLogger` method, which doubles as a factory function for Loggers
 * This is modeled after Python's logging module. When `getLogger` is called from
 * Python (via the pybind11 bindings) it creates a new Python-side Logger object
 * and registers its reference. When called C++ side it creates a C++-side
 * Logger and registers its reference. Accessing a `getLogger` using a name that
 * was already registered - returns the reference from the registry (python or
 * internal).
 *
 * The obvious pitfall is the case when a user does not route through this cls,
 * and instead registers a Python-side Logger via Python's logging module. Then
 * these Python-side Loggers are not registered in the Logging's registry the
 * KBMOD Logging will default to using the C++ std::out logger. This can lead to
 * differing output formats if the Python Logger in question is re-configured.
 */
namespace logging {
// Python's dict[str: str]-like typedef for readability
typedef std::unordered_map<std::string, std::string> sdict;

// translate between Python's log-levels in a way that will continue to
// respect any user-added in-between levels that are not necessarily
// registered in C++
// https://docs.python.org/3/library/logging.html#logging-levels
enum LogLevel { DEBUG = 10, INFO = 20, WARNING = 30, ERROR = 40, CRITICAL = 50 };

static std::unordered_map<std::string, LogLevel> StringToLogLevel{{"DEBUG", LogLevel::DEBUG},
                                                                  {"INFO", LogLevel::INFO},
                                                                  {"WARNING", LogLevel::WARNING},
                                                                  {"ERROR", LogLevel::ERROR},
                                                                  {"CRITICAL", LogLevel::CRITICAL}};

static std::unordered_map<LogLevel, std::string> LogLevelToString{{LogLevel::DEBUG, "DEBUG"},
                                                                  {LogLevel::INFO, "INFO"},
                                                                  {LogLevel::WARNING, "WARNING"},
                                                                  {LogLevel::ERROR, "ERROR"},
                                                                  {LogLevel::CRITICAL, "CRITICAL"}};

// Logger is a base class that dispatches the logging mechanism (IO mostly)
// It wraps convenience methods and other shared functionality, such as
// string formatters, commonly used by child Loggers. Expects the following
// config key-values to exist:
// `level`: `string``
//     LogLevel enum value, the minimal level that is printed.
// `datefmt`: `string`
//     Timestamp template usable with `std::put_time`.
// `format`: `string`
//     log format template, currently supports ``asctime``, ``levelname``,
//     ``names``, and ``message``. The fields in the string are expected to
//     be formatted as in Python ``%{field}s``
// `converter`: `string`
//     Time zone converter, either `gmtime` or `localtime`.
class Logger {
public:
    std::string name;
    sdict config;
    LogLevel level_threshold;

    Logger(const std::string logger_name) : name(logger_name), config(), level_threshold{LogLevel::WARNING} {}

    Logger(const std::string logger_name, const sdict conf) : name(logger_name), config(conf) {
        level_threshold = StringToLogLevel[config["level"]];
    }

    virtual ~Logger() {}

    std::string fmt_time() {
        std::time_t now = std::time(nullptr);
        std::tm timeinfo;

        if (config["converter"] == "gmtime") {
            timeinfo = *std::gmtime(&now);
        } else {
            timeinfo = *std::localtime(&now);
        }

        std::ostringstream timestamp;
        timestamp << std::put_time(&timeinfo, config["datefmt"].c_str());
        return timestamp.str();
    }

    std::string fmt_log(const std::string level, const std::string msg) {
        std::string logfmt = config["format"];

        std::regex t("%\\(asctime\\)s");
        logfmt = std::regex_replace(logfmt, t, fmt_time());

        std::regex l("%\\(levelname\\)s");
        logfmt = std::regex_replace(logfmt, l, level);

        std::regex n("%\\(name\\)s");
        logfmt = std::regex_replace(logfmt, n, name);

        std::regex m("%\\(message\\)s");
        logfmt = std::regex_replace(logfmt, m, msg);

        return logfmt;
    }

    virtual void log(std::string level, std::string msg) = 0;
    void debug(std::string msg) { log("DEBUG", msg); }
    void info(std::string msg) { log("INFO", msg); }
    void warning(std::string msg) { log("WARNING", msg); }
    void error(std::string msg) { log("ERROR", msg); }
    void critical(std::string msg) { log("CRITICAL", msg); }
};

// Glorified std::cout.
class CoutLogger : public Logger {
public:
    CoutLogger(std::string name, sdict config) : Logger(name, config) {}

    virtual void log(const std::string level, const std::string msg) {
        if (level_threshold <= StringToLogLevel[level]) std::cout << fmt_log(level, msg) << std::endl;
    }
};

// Wrapper around the Python-side loggers. Basically dispatches the logging
// calls to the Python-side object. Does no formatting, IO, or other management
// except to ensure the message is dispatched to the correct in-Python method.
#ifdef Py_PYTHON_H
class PyLogger : public Logger {
private:
    py::object pylogger;

public:
    PyLogger(py::object logger) : Logger(logger.attr("name").cast<std::string>()), pylogger(logger) {}

    virtual void log(std::string level, const std::string msg) {
        for (char& ch : level) ch = std::tolower(ch);
        pylogger.attr(level.c_str())(msg);
    }
};
#endif  // Py_PYTHON_H

// Logging is a singleton keeping the registry of all registered Loggers and
// their default configuration. Use `getLoger(name)` to get or create a new
// logger. When called, it will check if the logger exists and return a
// reference if it does. If it doesn't exists, and the method was called from
// Python's, it creates a new Python-side logger and returns it. When called
// from C++, without being able to provide a reference to a, or a name of an
// already existing, Python logger, it creates a default logger on the C++ side.
// This logger will share at leas the `logging.basicConfig`-uration with any
// already existing Python loggers. By default it will create a `CoutLogger`.
// If literally nothing exists, and KBMOD is being driven purely from C++ side
// it will instantiate a new default logger using the default configuration
// that matches the default Python logging configuration as closely as
// possible:
// - level:  "WARNING"
// - datefmt":  "%Y-%m-%dT%H:%M:%SZ"
// - converter": "localtime"
// - format: "[%(asctime)s %(levelname)s %(name)s] %(message)s"

class Logging {
private:
    sdict default_config = {{"level", "WARNING"},
                            {"datefmt", "%Y-%m-%dT%H:%M:%SZ"},
                            {"converter", "localtime"},
                            {"format", "[%(asctime)s %(levelname)s %(name)s] %(message)s"}};
    std::unordered_map<std::string, Logger*> registry;

    Logging() {}
    ~Logging() {
        for (auto elem = registry.begin(); elem != registry.end(); elem++) delete elem->second;
    }

public:
    // delete copy operators - it's a singleton
    Logging(Logging& other) = delete;
    void operator=(const Logging&) = delete;

    // get the singleton instance
    static Logging* logging() {
        static Logging instance;
        return &instance;
    }

    void setConfig(sdict config) { Logging::logging()->default_config = config; }

    sdict getConfig() { return Logging::logging()->default_config; }

    // Generic template to create any kind of new Logger instance and add it to
    // the registry at the same time. CamelCase to match the Python `logging`
    // module
    template <class LoggerCls>
    static Logger* getLogger(std::string name, sdict config = {}) {
        Logging* instance = Logging::logging();

        // if key not found use default setup
        if (instance->registry.find(name) == instance->registry.end()) {
            sdict tmpconf = config.size() != 0 ? config : instance->getConfig();
            instance->registry[name] = new LoggerCls(name, tmpconf);
        }
        return instance->registry[name];
    }

    static Logger* getLogger(std::string name, sdict config = {}) {
        return Logging::logging()->getLogger<CoutLogger>(name, config);
    }

    void register_logger(Logger* logger) { Logging::logging()->registry[logger->name] = logger; }
};

// Convenience method to shorten the very long signature required to invoke
// correct functionality: logging::Logging::logging()->getLogger(name)
// to logging::getLogger(name)
Logger* getLogger(std::string name, sdict config = {}) { return Logging::logging()->getLogger(name, config); }

#ifdef Py_PYTHON_H
static void logging_bindings(py::module& m) {
    py::class_<Logging, std::unique_ptr<Logging, py::nodelete>>(m, "Logging")
            .def(py::init([]() { return std::unique_ptr<Logging, py::nodelete>(Logging::logging()); }))
            .def("setConfig", &Logging::setConfig)
            .def_static("getLogger", [](py::str name) -> py::object {
                py::module_ logging = py::module_::import("logging");
                py::object pylogger = logging.attr("getLogger")(name);
                Logging::logging()->register_logger(new PyLogger(pylogger));
                return pylogger;
            });
}
#endif /* Py_PYTHON_H */
}  // namespace logging
#endif  // KBMOD_LOGGER
