"""
Logging configuration module for the vae model project.
Provides centralized logging setup with both file and console output.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Union, List
from rich.logging import RichHandler


class LoggerConfig:
    """
    Logger configuration class with customizable settings.

    Attributes:
        DEFAULT_FORMAT (str): Default log message format
        DEFAULT_DATE_FORMAT (str): Default date format for log messages
        DEFAULT_LEVEL (int): Default logging level
    """

    DEFAULT_FORMAT = "%(levelname)-7s - [%(asctime)s] %(lineno)-3d %(name)s - %(message)s"
    DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    DEFAULT_LEVEL = logging.INFO


def setup_logging(
    log_dir: Union[str, Path] = "logs",
    level: int = LoggerConfig.DEFAULT_LEVEL,
    log_format: str = LoggerConfig.DEFAULT_FORMAT,
    date_format: str = LoggerConfig.DEFAULT_DATE_FORMAT,
    filename_prefix: str = "",
    console_output: bool = False,
    handlers: Optional[List[logging.Handler]] = None,
) -> logging.Logger:
    """
    Configure logging with both file and console output.

    Args:
        log_dir (Union[str, Path]): Directory for log files
        level (int): Logging level (default: INFO)
        log_format (str): Format string for log messages
        date_format (str): Format string for timestamps
        filename_prefix (str): Prefix for log filename
        handlers (Optional[List[logging.Handler]]): Additional handlers

    Returns:
        logging.Logger: Configured root logger

    Raises:
        PermissionError: If unable to create log directory or file
        ValueError: If invalid logging configuration provided
    """
    try:
        # Convert log_dir to Path object and create directory
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Generate log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{filename_prefix}{timestamp}.log"

        # Create formatter
        formatter = logging.Formatter(log_format, date_format)

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(level)

        # Remove existing handlers
        root_logger.handlers.clear()

        # Add file handler
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        # Add console handler only if requested
        if console_output:
            # console_handler = logging.StreamHandler(sys.stdout)
            # console_handler.setFormatter(formatter)
            # root_logger.addHandler(console_handler)
            console_handler = RichHandler(
                markup=True,
                rich_tracebacks=True,
                show_time=False,
                show_level=False,
                show_path=False,
            )
            console_handler.setLevel(level)
            # Use your standard formatter for timestamps and other info
            console_formatter = logging.Formatter(log_format, date_format)
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)

        # Silence matplotlib and PIL debug messages
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("PIL").setLevel(logging.WARNING)

        # Log initial message
        root_logger.info(f"Logging configured - Log file: {log_file}")

        return root_logger

    except PermissionError as e:
        raise PermissionError(f"Unable to create log directory or file: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error configuring logger: {str(e)}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name (str): Logger name, typically __name__

    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)
