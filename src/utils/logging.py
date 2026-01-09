"""
Utility module for logging.

Usage:
    from src.utils.logging import Logger
    logger = Logger("outputs/train_log.txt")
    logger.log("Step 10, Loss: 2.5")
"""

from pathlib import Path
from typing import Optional


class Logger:
    """Simple logger that writes to both console and file."""

    def __init__(self, log_file: Optional[str] = None) -> None:
        """
        Initialize the logger.

        Args:
            log_file: Path to log file (optional)
        """
        self.log_file = Path(log_file) if log_file else None

        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str, print_to_console: bool = True) -> None:
        """
        Log a message.

        Args:
            message: Message to log
            print_to_console: Whether to print to console
        """
        if print_to_console:
            print(message)

        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(message + '\n')
