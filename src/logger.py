import logging
from pathlib import Path

from src.config import LOGS_DIR


# Create logs directory if it doesn't exist
LOGS_DIR.mkdir(exist_ok=True)

# Log file
LOG_FILE = LOGS_DIR / "project.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


