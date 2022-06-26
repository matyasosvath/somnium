using System;
using System.Collections.Generic;
using Codecool.Logging;

namespace Codecool.MainProduct.Logging
{
    /// <summary>
    /// An abstract implementation of the {@code Logger} interface.
    /// Holds some common logic.
    /// Extending classes should override the {@code safeLog} method which
    /// hides the filtering logic for the different log levels.
    /// </summary>
    public abstract class AbstractLogger : ILogger
    {
        private readonly HashSet<LogLevel> _levels;

        /// <summary>
        /// Initializes a new instance of the <see cref="AbstractLogger"/> class.
        /// </summary>
        /// <param name="levels">level parameters</param>
        public AbstractLogger(params LogLevel[] levels)
        {
            if (levels == null)
            {
                throw new ArgumentNullException(nameof(levels));
            }

            _levels = new HashSet<LogLevel>(levels);
        }

        /// <inheritdoc />
        public void Log(LogLevel level, string message)
        {
            WithLevel(level, () => SafeLog(level, message));
        }

        /// <summary>
        /// Log <paramref name="message"/> in a safe manner.
        /// </summary>
        /// <param name="level">Severity level.</param>
        /// <param name="message">Message to log.</param>
        protected abstract void SafeLog(LogLevel level, string message);


        /// <summary>
        /// Invoke <paramref name="logic"/> if the specified <paramref name="level"/> registered.
        /// </summary>
        /// <param name="level">Severity level.</param>
        /// <param name="logic">Actionable method</param>
        private void WithLevel(LogLevel level, Action logic)
        {
            if (_levels.Contains(level))
            {
                logic();
            }
        }
    }
}
