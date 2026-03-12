"""Extract text from images via OCR (handwritten logs, scanned documents)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.utils.text_cleaning import TextCleaner


def extract_ocr(image_path: Path, source_id: str, engine: str = "tesseract") -> dict:
    """Extract text from an image using OCR.

    Args:
        image_path: Path to image file (JPG, PNG, TIFF, etc.)
        source_id: Source identifier
        engine: OCR engine to use ('tesseract' or 'easyocr')
    """
    if engine == "easyocr":
        text = _extract_easyocr(image_path)
    else:
        text = _extract_tesseract(image_path)

    cleaned = TextCleaner.full_clean(text)
    sections = []

    if TextCleaner.is_meaningful_text(cleaned, min_words=5):
        sections.append({
            "title": image_path.stem,
            "text": cleaned,
            "metadata": {
                "content_type": "mixed",
                "ocr_engine": engine,
                "source_file": image_path.name,
            },
        })

    return {
        "source_id": source_id,
        "source_type": "image_ocr",
        "content_type": "mixed",
        "sections": sections,
    }


def _extract_tesseract(image_path: Path) -> str:
    """Extract text using Tesseract OCR."""
    import pytesseract
    from PIL import Image, ImageEnhance, ImageFilter

    img = Image.open(image_path)

    # Preprocessing for better OCR accuracy
    # Convert to grayscale
    img = img.convert("L")
    # Increase contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)
    # Sharpen
    img = img.filter(ImageFilter.SHARPEN)

    text = pytesseract.image_to_string(img, config="--psm 6")
    return text


def _extract_easyocr(image_path: Path) -> str:
    """Extract text using EasyOCR (better for handwritten text)."""
    import easyocr

    reader = easyocr.Reader(["en"], gpu=False)
    results = reader.readtext(str(image_path), detail=0)
    return "\n".join(results)


def extract_ocr_batch(image_dir: Path, source_id: str, engine: str = "tesseract") -> dict:
    """Extract text from all images in a directory."""
    image_extensions = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}
    sections = []

    image_files = sorted(
        f for f in image_dir.iterdir()
        if f.suffix.lower() in image_extensions
    )

    for img_path in image_files:
        result = extract_ocr(img_path, source_id, engine)
        sections.extend(result["sections"])

    return {
        "source_id": source_id,
        "source_type": "image_ocr",
        "content_type": "mixed",
        "sections": sections,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extract text from images via OCR")
    parser.add_argument("image_path", type=Path, help="Path to image file or directory")
    parser.add_argument("--source-id", required=True, help="Source identifier")
    parser.add_argument("--engine", choices=["tesseract", "easyocr"], default="tesseract")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    args = parser.parse_args()

    from scripts.utils.config import Config

    output_dir = args.output_dir or Config.EXTRACTED_DIR / "training_logs"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.image_path.is_dir():
        print(f"Extracting from directory: {args.image_path}")
        result = extract_ocr_batch(args.image_path, args.source_id, args.engine)
    else:
        print(f"Extracting: {args.image_path}")
        result = extract_ocr(args.image_path, args.source_id, args.engine)

    print(f"  Found {len(result['sections'])} sections")

    output_path = output_dir / f"{args.source_id}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()
