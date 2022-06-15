using System;
using System.Collections.Generic;
using System.Text;
using Codecool.Logging;

namespace Codecool.MainProduct.Logging
{
    /// <summary>
    /// An implementation of the <see cref="ILogger"/> interface.
    /// Logs messages for the standard error stream of the process.
    /// </summary>
    public class ConsoleLogger : AbstractLogger
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="ConsoleLogger"/> class.
        /// </summary>
        /// <param name="levels">Log levels.</param>
        public ConsoleLogger(params LogLevel[] levels) : base(levels) { }

        /// <inheritdoc />
        protected override void SafeLog(LogLevel level, string message)
        {
            string formattedMessage = FormatMessage(level, message);
            Console.WriteLine(formattedMessage);
        }

        private string FormatMessage(LogLevel level, string message)
        {
            var time = DateTime.Now.ToUniversalTime();
            return $"{time} - {level.ToString()}: {message} ";
        }
    }
}
