"""
Image preprocessing for OCR quality improvement.
Applies adaptive denoising, deskewing, and contrast enhancement
before passing images to Tesseract.
"""
import numpy as np
from loguru import logger
from typing import Tuple

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available — skipping advanced image preprocessing")

try:
    from PIL import Image, ImageEnhance, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def preprocess_image_for_ocr(pil_image: "Image.Image") -> "Image.Image":
    """
    Main entry point: apply full preprocessing pipeline to a PIL Image.
    Returns a cleaned PIL Image ready for Tesseract.
    """
    if not PIL_AVAILABLE:
        return pil_image

    # Convert to RGB if RGBA or palette mode
    if pil_image.mode in ("RGBA", "P", "LA"):
        pil_image = pil_image.convert("RGB")

    if CV2_AVAILABLE:
        return _preprocess_with_opencv(pil_image)
    else:
        return _preprocess_with_pillow(pil_image)


def _preprocess_with_opencv(pil_image: "Image.Image") -> "Image.Image":
    """Full OpenCV-based preprocessing pipeline."""
    # PIL → OpenCV (BGR)
    cv_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # 1. Convert to grayscale
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    # 2. Deskew
    gray = _deskew(gray)

    # 3. Denoise
    gray = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    # 4. Adaptive thresholding (handles uneven lighting on scanned pages)
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=10,
    )

    # 5. Morphological cleanup (remove small speckles)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # OpenCV grayscale → PIL
    return Image.fromarray(cleaned)


def _deskew(gray: np.ndarray) -> np.ndarray:
    """
    Detect and correct skew angle using Hough transform.
    Skips correction if angle is negligible (< 0.5°).
    """
    try:
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        # Probabilistic Hough lines
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=100, minLineLength=100, maxLineGap=10,
        )
        if lines is None:
            return gray

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if abs(angle) < 45:  # ignore near-vertical lines
                    angles.append(angle)

        if not angles:
            return gray

        median_angle = float(np.median(angles))
        if abs(median_angle) < 0.5:
            return gray  # not worth rotating

        logger.debug(f"Deskewing by {median_angle:.2f}°")
        h, w = gray.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(
            gray, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return rotated
    except Exception as e:
        logger.warning(f"Deskew failed: {e} — using original")
        return gray


def _preprocess_with_pillow(pil_image: "Image.Image") -> "Image.Image":
    """Fallback preprocessing using only Pillow (no OpenCV)."""
    # Grayscale
    img = pil_image.convert("L")
    # Sharpen
    img = img.filter(ImageFilter.SHARPEN)
    # Contrast boost
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)
    return img


def estimate_image_quality(pil_image: "Image.Image") -> float:
    """
    Return a rough quality score 0.0–1.0 for an image.
    Used to flag very low-quality pages in warnings.
    """
    if not CV2_AVAILABLE or not PIL_AVAILABLE:
        return 0.5

    gray = np.array(pil_image.convert("L"))
    # Laplacian variance — measures sharpness
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Normalize: 0 = blurry/empty, 1000+ = very sharp
    score = min(lap_var / 500.0, 1.0)
    return round(float(score), 3)
