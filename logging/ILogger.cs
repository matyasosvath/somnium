namespace Codecool.Logging
{
    /// <summary>
    /// A general interface for logging.
    /// All applications classes should be using one.
    /// </summary>
    public interface ILogger
    {
        /// <summary>
        /// Log a single event.
        /// The message will be logged only if the given log level is
        /// configured for the logger.
        /// </summary>
        /// <param name="level">The severity level of the event.</param>
        /// <param name="message">The message of the event.</param>
        void Log(LogLevel level, string message);
    }
}