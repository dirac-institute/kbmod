#ifndef DEBUG_TIMER_DOCS_
#define DEBUG_TIMER_DOCS_

namespace pydocs {
static const auto DOC_DEBUG_TIMER = R"doc(
  A simple timer used for consistent outputing of timing results in verbose mode.
  Timer automatically starts when it is created.

  Parameters
  ----------
  name : `str`
      The name string of the timer. Used for ouput.
  verbose : `bool`
      Output the timing information to the standard output.
  )doc";

static const auto DOC_DEBUG_TIMER_start = R"doc(
  Start (or restart) the timer. If verbose outputs a message.
  )doc";

static const auto DOC_DEBUG_TIMER_stop = R"doc(
  Stop the timer. If verbose outputs the duration.
  )doc";

static const auto DOC_DEBUG_TIMER_read = R"doc(
  Read the timer duration as decimal seconds.

  Returns
  -------
  duration : `float`
      The duration of the timer.
  )doc";

}  // namespace pydocs

#endif /* #define DEBUG_TIMER_DOCS_ */
