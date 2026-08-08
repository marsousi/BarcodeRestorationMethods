from abc import ABC, abstractmethod
from pathlib import Path

from barcode_restoration.result import RestorationResult


class RestorationMethod(ABC):
    """Common interface implemented by every restoration method."""

    name: str

    @abstractmethod
    def restore(
        self,
        input_path: Path,
        output_path: Path,
        *,
        variant: str | None = None,
        device: str = "cuda",
    ) -> RestorationResult:
        """Restore one image and return information about the result."""
        raise NotImplementedError
