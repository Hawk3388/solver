"""Public WorksheetSolver facade and command-line entry point."""

import os
from pathlib import Path
import re
import tempfile

from dotenv import load_dotenv
from google import genai
import requests

from worksheet_solver.batch import BatchResult, collect_input_images, solve_batch
from worksheet_solver.config import (
    DEFAULT_MODEL, DEFAULT_LOCAL, DEFAULT_THINK,
    REQUEST_TIMEOUT_SECONDS, solver_defaults,
)
from worksheet_solver.detection import DetectionMixin
from worksheet_solver.images import (
    MAX_OUTPUT_PIXELS,
    normalize_worksheet_image,
)
from worksheet_solver.rendering import RenderingMixin
from worksheet_solver.runtime import (
    DetectionModelRuntime,
    get_detection_model_runtime,
)
from worksheet_solver.schemas import Pair, get_solution
from worksheet_solver.solving import SolvingMixin


class WorksheetSolver(DetectionMixin, SolvingMixin, RenderingMixin):
    """Coordinate worksheet detection, solving, and answer rendering."""

    def __init__(
        self,
        path: str,
        gap_detection_model_path: str = "",
        llm_model_name: str = DEFAULT_MODEL,
        think: bool = DEFAULT_THINK,
        local: bool = DEFAULT_LOCAL,
        debug: bool = False,
        detection_runtime: DetectionModelRuntime | None = None,
        max_output_pixels: int = MAX_OUTPUT_PIXELS,
        auto_rotate_page: bool = True,
        correct_perspective: bool = True,
    ):
        self.debug = debug
        if not isinstance(llm_model_name, str) or not llm_model_name.strip():
            raise ValueError("Model name must not be empty.")
        self.model_name = llm_model_name.strip()
        self.local = local
        self.path = str(path)
        self.think = think
        self.max_output_pixels = max_output_pixels
        self.auto_rotate_page = auto_rotate_page
        self.correct_perspective = correct_perspective
        self.original_path = str(Path(path).expanduser().resolve())
        self.answer_warnings = []
        self.render_warnings = []
        self.normalization_info = {}

        if detection_runtime is not None:
            if gap_detection_model_path and (
                Path(gap_detection_model_path).expanduser().resolve()
                != detection_runtime.model_path
            ):
                raise ValueError(
                    "gap_detection_model_path does not match the shared "
                    "detection runtime."
                )
            self.model_path = str(detection_runtime.model_path)
        else:
            self.model_path = (
                gap_detection_model_path
                if gap_detection_model_path
                else self.get_gap_model()
            )
        self.image = None
        self.allowed_extensions = {"png", "jpg", "jpeg", "webp", "bmp"}
        self.detected_gaps = []
        self.gap_groups = []
        self.gap_to_group = {}
        self.ungrouped_gap_indices = []
        self.answer_units = []
        self.gap_to_answer_unit = {}
        self._temporary_directory = tempfile.TemporaryDirectory(
            prefix="worksheet_solver_"
        )
        self.temporary_directory = Path(self._temporary_directory.name)

        if self.debug:
            import time

            self.time = time

        try:
            self._prepare_input_image()
            if not Path(self.model_path).exists():
                raise FileNotFoundError(
                    f"Gap detection model not found: {self.model_path}"
                )

            if not self.local:
                self._configure_cloud_client()

            self.detection_runtime = (
                detection_runtime
                if detection_runtime is not None
                else get_detection_model_runtime(self.model_path)
            )
            # Kept as a compatibility alias for callers that inspect the model.
            self.model = self.detection_runtime.model
        except Exception:
            self.close()
            raise

    def __enter__(self):
        """Use the solver as a resource-owning context manager."""
        return self

    def __exit__(self, _exception_type, _exception, _traceback):
        self.close()
        return False

    def __del__(self):
        """Best-effort fallback for callers that do not use a context manager."""
        try:
            self.close()
        except Exception:
            pass

    def close(self):
        """Release every temporary artifact owned by this solver."""
        temporary_directory = getattr(self, "_temporary_directory", None)
        if temporary_directory is None:
            return
        temporary_directory.cleanup()
        self._temporary_directory = None

    def temporary_path(self, filename: str) -> Path:
        """Return a safe path inside this solver's temporary directory."""
        if self._temporary_directory is None:
            raise RuntimeError("The worksheet solver has already been closed.")
        return self.temporary_directory / Path(filename).name

    def _prepare_input_image(self):
        """Validate and normalize the source into a temporary RGB PNG."""
        source_path = Path(self.path)
        if not source_path.exists():
            raise FileNotFoundError(f"Worksheet image not found: {self.path}")
        if not self.is_allowed_image(self.path):
            allowed = ", ".join(sorted(self.allowed_extensions))
            raise ValueError(f"Invalid file type. Allowed types: {allowed}")
        converted_path = self.temporary_path(
            f"{source_path.stem}_normalized.png"
        )
        normalize_worksheet_image(
            source_path,
            converted_path,
            max_output_pixels=self.max_output_pixels,
            auto_rotate_page=self.auto_rotate_page,
            correct_perspective=self.correct_perspective,
            metadata=self.normalization_info,
        )

        self.path = str(converted_path)

    def _configure_cloud_client(self):
        """Create the Gemini client used by OCR and worksheet solving."""
        if os.path.exists(".env"):
            load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Google API key not found. Set GOOGLE_API_KEY in the "
                "environment or .env file."
            )
        from google.genai import types
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_SECONDS * 1000),
        )

    @classmethod
    def resolve_gap_model(cls, debug: bool = False) -> str:
        """Return an installed detector model or download it on first use."""
        releases_url = "https://github.com/Hawk3388/solver/releases"
        model_directory = Path("./model")
        model_directory.mkdir(parents=True, exist_ok=True)
        version_pattern = re.compile(r"^v\d+\.\d+\.\d+$")
        installed_versions = [
            path.name
            for path in model_directory.iterdir()
            if path.is_dir() and version_pattern.match(path.name)
        ]
        installed_versions.sort(
            key=lambda value: tuple(map(int, value.lstrip("v").split("."))),
            reverse=True,
        )

        for version in installed_versions:
            candidate = model_directory / version / "gap_detection_model.pt"
            if candidate.exists():
                return str(candidate)

        try:
            release_response = requests.get(releases_url, timeout=8)
            release_response.raise_for_status()
            pattern = re.compile(r"<h2[^>]*>(v\d+\.\d+\.\d+)</h2>")
            remote_versions = pattern.findall(release_response.text)
            remote_versions.sort(
                key=lambda value: tuple(map(int, value.lstrip("v").split("."))),
                reverse=True,
            )

            for version in remote_versions:
                model_url = (
                    "https://github.com/Hawk3388/solver/releases/download/"
                    f"{version}/gap_detection_model.pt"
                )
                if not cls.url_exists(model_url):
                    continue

                target = model_directory / version / "gap_detection_model.pt"
                temporary_target = target.with_suffix(".pt.part")
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with requests.get(
                        model_url,
                        stream=True,
                        timeout=60,
                    ) as response:
                        response.raise_for_status()
                        with open(temporary_target, "wb") as model_file:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    model_file.write(chunk)
                    temporary_target.replace(target)
                finally:
                    temporary_target.unlink(missing_ok=True)
                return str(target)
        except requests.RequestException as error:
            if debug:
                print(f"Model update check skipped: {error}")

        raise FileNotFoundError(
            "No gap detection model is installed and the latest model could "
            "not be downloaded."
        )

    def get_gap_model(self) -> str:
        """Resolve the detector weights configured for this solver."""
        return self.resolve_gap_model(debug=self.debug)

    @classmethod
    def preload_detection_model(
        cls,
        model_path: str = "",
        *,
        debug: bool = False,
    ) -> DetectionModelRuntime:
        """Load and cache the detector before worksheet requests arrive."""
        resolved_path = model_path or cls.resolve_gap_model(debug=debug)
        return get_detection_model_runtime(resolved_path)

    @staticmethod
    def url_exists(url: str, timeout: float = 5.0) -> bool:
        """Return whether a remote model URL is available."""
        try:
            response = requests.head(
                url,
                allow_redirects=True,
                timeout=timeout,
            )
            return 200 <= response.status_code < 400
        except requests.RequestException:
            return False

    def is_allowed_image(self, filename: str) -> bool:
        """Validate an input filename against supported image extensions."""
        return (
            "." in filename
            and filename.rsplit(".", 1)[1].lower() in self.allowed_extensions
        )


def main():
    """Run the interactive command-line workflow."""
    raw_input = input(
        "Enter an image, a directory, or semicolon-separated paths: "
    ).strip()
    solver = None
    try:
        requested_paths = [
            entry.strip().strip('"')
            for entry in raw_input.split(";")
            if entry.strip()
        ]
        if not requested_paths:
            print("No input selected.")
            return

        images = collect_input_images(requested_paths)
        if not images:
            print("No supported images found.")
            return

        is_batch = len(images) > 1 or any(
            Path(entry).expanduser().is_dir() for entry in requested_paths
        )
        if is_batch:
            output_directory = Path.cwd() / "solved_batch"
            print(f"Processing {len(images)} worksheet(s) sequentially...")
            results = solve_batch(
                images,
                output_directory,
                solver_kwargs=solver_defaults(),
                solver_class=WorksheetSolver,
            )
            successful = [result for result in results if result.success]
            for result in results:
                if result.success:
                    print(f"  OK     {result.source_path.name} -> {result.output_path}")
                else:
                    print(f"  FAILED {result.source_path.name}: {result.error}")
            print(
                f"Finished: {len(successful)}/{len(results)} worksheet(s) solved."
            )
            return

        path = str(images[0])
        solver = WorksheetSolver(
            path,
            **solver_defaults(),
            debug=True,
        )
        print("Loading image and detecting gaps...")
        gaps, image = solver.detect_gaps()
        print(
            f"{len(gaps)} boxes found, {len(solver.gap_groups)} line groups, "
            f"{len(solver.ungrouped_gap_indices)} ungrouped!"
        )
        marked_image = solver.mark_gaps(image, gaps)

        print("\nDetected gaps (x1, y1, x2, y2, class):")
        for index, gap in enumerate(gaps):
            unit_number = solver.gap_to_answer_unit.get(index)
            if unit_number is None:
                print(f"  Box {index + 1} (ungrouped): {gap}")
            else:
                print(f"  Box {index + 1} (Group {unit_number + 1}): {gap}")

        print("\nGap groups:")
        for group_index, group in enumerate(solver.gap_groups):
            print(
                f"  Group {group_index + 1}: "
                f"gaps {[index + 1 for index in group]}"
            )

        user_input = input(
            "\nShould an AI analyze and fill the gaps? (y/N): "
        ).lower().strip()
        if user_input not in {"y", "yes"}:
            print("Gap detection only")
            return

        solutions = solver.solve_all_gaps(marked_image)
        if not solutions:
            print("No solutions received.")
            return

        print("\nSolutions found:")
        for group_index, solution in solutions.items():
            gap_indices = [index + 1 for index in solution["gap_indices"]]
            print(
                f"  Group {group_index + 1} (gaps {gap_indices}): "
                f"'{solution['solution']}'"
            )
        solver.fill_gaps_in_image(solver.path, solutions)
        print("\nResult saved.")
    except FileNotFoundError as error:
        print(f"Error: {error}")
    except Exception as error:
        print(f"Unexpected error: {error}")
    finally:
        if solver is not None:
            try:
                solver.close()
            except OSError as error:
                print(f"Warning: temporary files could not be removed: {error}")


if __name__ == "__main__":
    main()
