from datetime import datetime
from typing import Optional

from src.core.consts import FORMATS


def create_run_id(run_id: Optional[str]) -> str:
    return run_id if (run_id is not None) else datetime.now().strftime(FORMATS.TIME_WITH_MICROSECONDS)
