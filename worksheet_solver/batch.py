"""Batch orchestration for solving multiple worksheet images."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass(frozen=True)
class BatchResult:
    """Outcome of processing one worksheet in a batch."""

    source_path: Path
    output_path: Path | None = None
    error: str | None = None

    @property
    def success(self) -> bool:
        """Return whether a solved image was created."""
        return self.output_path is not None and self.error is None


def collect_input_images(
    inputs: Iterable[str | Path] | str | Path,
    *,
    recursive: bool = False,
) -> list[Path]:
    """Expand image files and directories into a stable, deduplicated list."""
    if isinstance(inputs, (str, Path)):
        inputs = [inputs]

    images: list[Path] = []
    seen: set[Path] = set()

    for raw_input in inputs:
        path = Path(raw_input).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Input does not exist: {path}")

        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            iterator = path.rglob("*") if recursive else path.iterdir()
            candidates = sorted(
                (candidate for candidate in iterator if candidate.is_file()),
                key=lambda candidate: str(candidate).lower(),
            )
        else:
            continue

        for candidate in candidates:
            if candidate.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                continue
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                images.append(resolved)

    return images


def _available_output_path(output_directory: Path, source: Path) -> Path:
    """Return a non-existing output path without overwriting earlier results."""
    candidate = output_directory / f"{source.stem}_solved.png"
    suffix = 2
    while candidate.exists():
        candidate = output_directory / f"{source.stem}_solved_{suffix}.png"
        suffix += 1
    return candidate


def solve_batch(
    inputs: Iterable[str | Path] | str | Path,
    output_directory: str | Path,
    *,
    solver_kwargs: dict[str, Any] | None = None,
    recursive: bool = False,
    continue_on_error: bool = True,
    solver_class=None,
) -> list[BatchResult]:
    """Solve images sequentially and return one result record per image.

    Sequential processing avoids loading several detection and language models at
    once. A failed worksheet does not stop the remaining batch unless
    ``continue_on_error`` is disabled.
    """
    image_paths = collect_input_images(inputs, recursive=recursive)
    destination = Path(output_directory).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    if solver_class is None:
        # Imported lazily to keep this module independent from the public facade.
        from main import WorksheetSolver

        solver_class = WorksheetSolver

    options = dict(solver_kwargs or {})
    if "path" in options:
        raise ValueError("Pass input paths to solve_batch(), not in solver_kwargs.")

    results: list[BatchResult] = []
    for source_path in image_paths:
        solver = None
        try:
            solver = solver_class(str(source_path), **options)
            try:
                gaps, image = solver.detect_gaps()
                if not gaps:
                    raise RuntimeError("No gaps detected in the worksheet.")

                marked_image = solver.mark_gaps(image, gaps)
                solutions = solver.solve_all_gaps(marked_image)
                if not solutions:
                    raise RuntimeError("The AI could not find any solutions.")

                output_path = _available_output_path(destination, source_path)
                solver.fill_gaps_in_image(
                    solver.path,
                    solutions,
                    output_path=str(output_path),
                )
                if not output_path.is_file():
                    raise RuntimeError(
                        "The renderer did not create an output image."
                    )
            finally:
                close = getattr(solver, "close", None)
                if close is not None:
                    close()
            results.append(BatchResult(source_path, output_path=output_path))
        except Exception as error:
            results.append(BatchResult(source_path, error=str(error)))
            if not continue_on_error:
                raise

    return results
