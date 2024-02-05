#ifndef KBMOD_LOGGER
#define KBMOD_LOGGER

#include <iostream>
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
  // There must be an easier way to establish mapping between log level and str
  // but I don't know it. Advice appreciated. Also switch statement? pattern
  // matching? Uppercase/lowercase? string isn't a type? what year are we in?

  // https://docs.python.org/3/library/logging.html#logging-levels
  enum LogLevel {
    DEBUG = 10,
    INFO = 20,
    WARNING = 30,
    ERROR = 40,
    CRITICAL = 50
  };

  static std::unordered_map<std::string, LogLevel> StringToLogLevel {
    {"DEBUG",    LogLevel::DEBUG},
    {"INFO",     LogLevel::INFO},
    {"WARNING",  LogLevel::WARNING},
    {"ERROR",    LogLevel::ERROR},
    {"CRITICAL", LogLevel::CRITICAL}
  };

  static std::unordered_map<LogLevel, std::string> LogLevelToString {
    {LogLevel::DEBUG,    "DEBUG"},
    {LogLevel::INFO,     "INFO"},
    {LogLevel::WARNING,  "WARNING"},
    {LogLevel::ERROR,    "ERROR"},
    {LogLevel::CRITICAL, "CRITICAL"}
  };


  // dictionary of strings are used to configure Logger behavior such as minimal
  // log level allowed to print, time-format, log format etc.
  typedef std::unordered_map<std::string, std::string> sdict;

  // Logger is a base class that dispatches the logging mechanism (IO and
  // formatting) via the virtual method `log`
  class Logger {
  public:
    std::string name;
    sdict config;
    LogLevel level_threshold;

    Logger(std::string logger_name) :
      name(logger_name), config(), level_threshold{LogLevel::WARNING}
    {}

    Logger(std::string logger_name, sdict conf) :
      name(logger_name), config(conf) {
      level_threshold = StringToLogLevel[config["level"]];
    }

    virtual void log(std::string level, std::string msg) = 0;
    void debug(std::string msg) { log("DEBUG", msg); }
    void info(std::string msg) { log("INFO", msg); }
    void warning(std::string msg) { log("WARNING", msg); }
    void error(std::string msg) { log("ERROR", msg); }
    void critical(std::string msg) { log("CRITICAL", msg); }
  };


#ifdef Py_PYTHON_H
  // Passes the logging to the Python-side Logger. The py::handle is the
  // pybind11 container holding the reference to the Python Logger. The actual
  // logging format and mechanism is handled Python-side by the logging module,
  // so just pass the message and level onward and let it be resolved in Python
  class PyLogger : public Logger {
  private:
    py::handle pylogger;

  public:
    PyLogger(py::handle logger) :
      Logger(logger.attr("name").cast<std::string>()),
      pylogger(logger)
    {}

    virtual void log(std::string level, std::string msg){
      for (char& ch : level) ch = std::tolower(ch);
      pylogger.attr(level.c_str())(msg);
    }
  };
#endif // Py_PYTHON_H


  // Glorified std::cout. Unlike the PyLogger, CoutLogger is configurable.
  // Configuration should contain `level`, `datefmt` and `format` keys:
  // - `level`:  LogLevel enum value, the minimal level that is printed
  // - `datefmt`: timestamp template usable with `strftime`
  // - `format`: log format template, too much commitment atm, not supported
  class CoutLogger : public Logger{
  private:
    // datefmt %Y-%m expands to YYYY-MM etc., 2x its size sounds safe enough but
    // with abuse I guess this could overflow?
    std::string fmt_time(){
      time_t now;
      time(&now);
      char buf[2*sizeof(config["datefmt"])] = {0};
      strftime(buf, sizeof buf, config["datefmt"].c_str(), gmtime(&now));
      return std::string(buf);
    }

  public:
    CoutLogger(std::string name, sdict config) :
      Logger(name, config)
    {}

    virtual void log(std::string level, std::string msg){
      if (level_threshold <= StringToLogLevel[level])
        std::cout << "[" << fmt_time() << " " << level << " " << name << "] " << msg << std::endl;
    }
  };


  // The singleton keeping the registry of all registered Loggers and their
  // default configuration. Use `logger` to access the singleton instance,
  // getLoger to get an existing logger or to create a new default logger.
  // No good examples, but let's say a non-stdout C++ only logger is required,
  // use getLogger<LoggerCls>(name, conf) to register and get the instance of it
  class Logging{
  private:
    static sdict default_config;
    std::map<std::string, Logger*> registry;
    static Logging* instance;

    Logging(){}
    ~Logging(){}

  public:
    Logging(Logging &other) = delete;
    void operator=(const Logging &) = delete;

    // get the singleton instance
    static Logging* logger(){
      if(instance == nullptr)
        instance = new Logging();
      return instance;
    }

    static void setConfig(sdict config){
      default_config = config;
    }

    template<class LoggerCls>
    static Logger* getLogger(std::string name, sdict config={}){
      if (instance == nullptr)
        instance = new Logging();

      // if key not found use default setup
      if (instance->registry.find(name) == instance->registry.end()) {
        sdict tmpconf = config.size() != 0 ? config : instance->default_config;
        instance->registry[name] = new LoggerCls(name, tmpconf);
      }
      return instance->registry[name];
    }

    static Logger* getLogger(std::string name, sdict config={}){
      return getLogger<CoutLogger>(name, config);
    }

    void register_logger(Logger* logger){
      instance->registry[logger->name] = logger;
    }
  };

  Logging* Logging::instance = nullptr;
  sdict Logging::default_config = {
    {"level", "WARNING"},
    {"datefmt", "'%Y-%m-%dT%H:%M:%SZ"}
  };

  // This is for convenience sake in C++ code to
  // shorten logging::Logger::getLogger(name) to
  // logging::getLogger(name)
  Logger* getLogger(std::string name, sdict config={}){
    return Logging::getLogger(name, config);
  }


#ifdef Py_PYTHON_H
  static void logging_bindings(py::module& m) {
    py::class_<Logging, std::unique_ptr<Logging, py::nodelete>>(m, "Logging")
      .def(py::init([](){ return std::unique_ptr<Logging, py::nodelete>(Logging::logger()); }))
      .def("setConfig", &Logging::setConfig)
      .def("getLogger", [](py::str name) -> py::handle {
        py::module_ logging = py::module_::import("logging");
        py::handle pylogger = logging.attr("getLogger")(name);
        Logging::logger() -> register_logger(new PyLogger(pylogger));
        return pylogger;
      });
  }
#endif /* Py_PYTHON_H */
} // namespace logging
#endif // KBMOD_LOGGER
