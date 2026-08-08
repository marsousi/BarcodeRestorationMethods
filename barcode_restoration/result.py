from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class RestorationResult:
    method: str
    input_path: Path
    output_path: Path
    elapsed_seconds: float
    metadata: dict[str, Any]
