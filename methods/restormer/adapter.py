from pathlib import Path
import shutil
import subprocess
import tempfile
import time

from barcode_restoration.base import RestorationMethod
from barcode_restoration.registry import register_method
from barcode_restoration.result import RestorationResult


@register_method
class RestormerMethod(RestorationMethod):
    """Adapter for the official Restormer repository."""

    name = "restormer"

    SUPPORTED_VARIANTS = {
        "motion_deblur": "Motion_Deblurring",
    }

    def restore(
        self,
        input_path: Path,
        output_path: Path,
        *,
        variant: str | None = None,
        device: str = "cuda",
    ) -> RestorationResult:
        if variant is None:
            variant = "motion_deblur"

        if variant not in self.SUPPORTED_VARIANTS:
            supported = ", ".join(sorted(self.SUPPORTED_VARIANTS))
            raise ValueError(
                f"Unsupported Restormer variant '{variant}'. "
                f"Supported variants: {supported}"
            )

        if device != "cuda":
            raise ValueError(
                "The currently integrated official Restormer demo "
                "supports CUDA inference only."
            )

        method_dir = Path(__file__).resolve().parent
        upstream_dir = method_dir / "upstream"
        python_executable = method_dir / ".venv" / "bin" / "python"
        demo_script = upstream_dir / "demo.py"

        checkpoint = (
            upstream_dir
            / "Motion_Deblurring"
            / "pretrained_models"
            / "motion_deblurring.pth"
        )

        required_paths = {
            "Restormer upstream repository": upstream_dir,
            "Restormer Python environment": python_executable,
            "Restormer demo": demo_script,
            "Restormer checkpoint": checkpoint,
        }

        for description, path in required_paths.items():
            if not path.exists():
                raise FileNotFoundError(
                    f"{description} was not found: {path}"
                )

        input_path = input_path.resolve()
        output_path = output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        task = self.SUPPORTED_VARIANTS[variant]

        start = time.perf_counter()

        with tempfile.TemporaryDirectory(
            prefix="restormer_"
        ) as temporary_directory:
            result_root = Path(temporary_directory)

            command = [
                str(python_executable),
                str(demo_script),
                "--task",
                task,
                "--input_dir",
                str(input_path),
                "--result_dir",
                str(result_root),
            ]

            subprocess.run(
                command,
                cwd=upstream_dir,
                check=True,
            )

            restored_path = (
                result_root
                / task
                / f"{input_path.stem}.png"
            )

            if not restored_path.is_file():
                raise RuntimeError(
                    "Restormer completed but the expected restored "
                    f"image was not created: {restored_path}"
                )

            shutil.copy2(restored_path, output_path)

        elapsed = time.perf_counter() - start

        return RestorationResult(
            method=self.name,
            input_path=input_path,
            output_path=output_path,
            elapsed_seconds=elapsed,
            metadata={
                "variant": variant,
                "task": task,
                "device": device,
                "upstream_commit": (
                    "68dc6ac472db26f16361150cb7a96a1bc87da93f"
                ),
                "checkpoint_sha256": (
                    "194e38fb5b607c9dc5a5b3e08e65b2e79ee2bf0ef5048e0612f6b2ff2f79da31"
                ),
            },
        )
