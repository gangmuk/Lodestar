import logging
import os

INCLUDE_GPU_IN_FEATURE = False


# Configure logging
def setup_logging(log_file_path=None):
    logging_level = os.environ.get("LOG_LEVEL", "INFO")

    # Add %(filename)s to the format string to include the file name
    # print function name as well
    handlers = [logging.StreamHandler()]

    # Add file handler if log_file_path is provided
    if log_file_path:
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        handlers.append(logging.FileHandler(log_file_path))

    logging.basicConfig(
        level=logging_level,
        # format="%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s:%(lineno)d - %(message)s",
        format="%(filename)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
        handlers=handlers,
        force=True  # Allow reconfiguration if called multiple times
    )

    # logging.basicConfig(level=getattr(logging, logging_level), format='%(asctime)s - %(levelname)s - %(filename)s - %(message)s')



    logger = logging.getLogger("llm_router")
    return logger

# Create and export a common logger instance
logger = setup_logging()