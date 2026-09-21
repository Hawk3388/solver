"""Safe worksheet image validation and normalization."""

import math
from pathlib import Path
import warnings

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


MAX_SOURCE_PIXELS = 40_000_000
MAX_OUTPUT_PIXELS = 16_000_000
MAX_ASPECT_RATIO = 6.0


class ImageNormalizationError(ValueError):
    """Raised when an image is invalid or unsafe to decode."""


def _validate_dimensions(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise ImageNormalizationError("The image has invalid dimensions.")

    pixel_count = width * height
    if pixel_count > MAX_SOURCE_PIXELS:
        raise ImageNormalizationError(
            f"The image contains {pixel_count:,} pixels; the safety limit is "
            f"{MAX_SOURCE_PIXELS:,} pixels."
        )

    aspect_ratio = max(width, height) / min(width, height)
    if aspect_ratio > MAX_ASPECT_RATIO:
        raise ImageNormalizationError(
            f"The image aspect ratio is {aspect_ratio:.1f}:1; ratios above "
            f"{MAX_ASPECT_RATIO:.1f}:1 are not accepted."
        )


def validate_image_file(image_path: str | Path) -> tuple[int, int]:
    """Validate image headers and content without retaining an open handle."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(image_path) as image:
                _validate_dimensions(*image.size)
                dimensions = image.size
                image.verify()
                return dimensions
    except ImageNormalizationError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise ImageNormalizationError(
            "The image exceeds Pillow's decompression safety limit."
        ) from error
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as error:
        raise ImageNormalizationError(
            "The uploaded file is not a valid supported image."
        ) from error


def _flatten_to_rgb(image: Image.Image) -> Image.Image:
    """Convert every Pillow mode to opaque RGB using a white background."""
    has_transparency = image.mode in {"RGBA", "LA"} or (
        image.mode == "P" and "transparency" in image.info
    )
    if not has_transparency:
        return image.convert("RGB")

    rgba_image = image.convert("RGBA")
    try:
        background = Image.new("RGB", rgba_image.size, (255, 255, 255))
        try:
            alpha_channel = rgba_image.getchannel("A")
            try:
                background.paste(rgba_image, mask=alpha_channel)
            finally:
                alpha_channel.close()
            return background
        except Exception:
            background.close()
            raise
    finally:
        rgba_image.close()


def _correct_page_perspective(image: Image.Image) -> Image.Image | None:
    """Return a conservative four-corner page warp, if one is evident."""
    pixels = np.asarray(image)
    grayscale = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None

    image_area = image.width * image.height
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        if cv2.contourArea(contour) < image_area * 0.45:
            continue
        perimeter = cv2.arcLength(contour, True)
        corners = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(corners) != 4:
            continue

        points = corners.reshape(4, 2).astype("float32")
        point_sums = points.sum(axis=1)
        point_differences = np.diff(points, axis=1).reshape(-1)
        ordered = np.array(
            [
                points[np.argmin(point_sums)],
                points[np.argmin(point_differences)],
                points[np.argmax(point_sums)],
                points[np.argmax(point_differences)],
            ],
            dtype="float32",
        )
        top_left, top_right, bottom_right, bottom_left = ordered
        target_width = int(
            max(
                np.linalg.norm(bottom_right - bottom_left),
                np.linalg.norm(top_right - top_left),
            )
        )
        target_height = int(
            max(
                np.linalg.norm(top_right - bottom_right),
                np.linalg.norm(top_left - bottom_left),
            )
        )
        if target_width < 64 or target_height < 64:
            continue
        _validate_dimensions(target_width, target_height)

        destination = np.array(
            [
                [0, 0],
                [target_width - 1, 0],
                [target_width - 1, target_height - 1],
                [0, target_height - 1],
            ],
            dtype="float32",
        )
        transform = cv2.getPerspectiveTransform(ordered, destination)
        warped = cv2.warpPerspective(
            pixels,
            transform,
            (target_width, target_height),
            borderMode=cv2.BORDER_REPLICATE,
        )
        return Image.fromarray(warped)
    return None


def normalize_worksheet_image(
    source_path: str | Path,
    destination_path: str | Path,
    *,
    max_output_pixels: int = MAX_OUTPUT_PIXELS,
    auto_rotate_page: bool = False,
    correct_perspective: bool = False,
) -> Path:
    """Normalize a worksheet to an oriented, opaque, bounded RGB PNG."""
    if max_output_pixels <= 0 or max_output_pixels > MAX_SOURCE_PIXELS:
        raise ValueError(
            f"max_output_pixels must be between 1 and {MAX_SOURCE_PIXELS:,}."
        )

    destination = Path(destination_path)
    resources: list[Image.Image] = []

    def own(image: Image.Image, source: Image.Image) -> Image.Image:
        if image is not source:
            resources.append(image)
        return image

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(source_path) as source_image:
                _validate_dimensions(*source_image.size)
                dpi = source_image.info.get("dpi")
                source_image.load()

                oriented = own(ImageOps.exif_transpose(source_image), source_image)
                normalized = own(_flatten_to_rgb(oriented), oriented)

                if correct_perspective:
                    corrected = _correct_page_perspective(normalized)
                    if corrected is not None:
                        resources.append(corrected)
                        normalized = corrected

                if auto_rotate_page and normalized.width > normalized.height:
                    rotated = normalized.transpose(Image.Transpose.ROTATE_90)
                    resources.append(rotated)
                    normalized = rotated

                pixel_count = normalized.width * normalized.height
                if pixel_count > max_output_pixels:
                    scale = math.sqrt(max_output_pixels / pixel_count)
                    target_size = (
                        max(1, int(normalized.width * scale)),
                        max(1, int(normalized.height * scale)),
                    )
                    resized = normalized.resize(
                        target_size,
                        Image.Resampling.LANCZOS,
                    )
                    resources.append(resized)
                    normalized = resized

                _validate_dimensions(*normalized.size)
                save_options = {"format": "PNG", "optimize": True}
                if dpi:
                    save_options["dpi"] = dpi
                normalized.save(destination, **save_options)
    except ImageNormalizationError:
        destination.unlink(missing_ok=True)
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        destination.unlink(missing_ok=True)
        raise ImageNormalizationError(
            "The image exceeds Pillow's decompression safety limit."
        ) from error
    except (UnidentifiedImageError, OSError, SyntaxError) as error:
        destination.unlink(missing_ok=True)
        raise ImageNormalizationError(
            "The image could not be decoded or normalized."
        ) from error
    finally:
        closed: set[int] = set()
        for resource in reversed(resources):
            if id(resource) not in closed:
                resource.close()
                closed.add(id(resource))

    return destination
