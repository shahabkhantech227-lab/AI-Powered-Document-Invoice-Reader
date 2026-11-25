from pathlib import Path
import argparse
import csv
import json

import cv2
import numpy as np
import pytesseract
from pytesseract import Output


def deskew(gray):
    coords = np.column_stack(np.where(gray < 250))
    if len(coords) < 20:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) > 15:
        return gray
    h, w = gray.shape[:2]
    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray, matrix, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def preprocess(path):
    original = cv2.imread(str(path))
    if original is None:
        raise FileNotFoundError(f"Cannot open image: {path}")
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    straight = deskew(blurred)
    binary = cv2.adaptiveThreshold(
        straight, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 11
    )
    return original, gray, binary


def recognize(binary, threshold):
    data = pytesseract.image_to_data(
        binary, config="--oem 3 --psm 6", output_type=Output.DICT
    )
    words = []
    for i, raw in enumerate(data["text"]):
        word = raw.strip()
        try:
            conf = float(data["conf"][i])
        except (ValueError, TypeError):
            continue
        if word and conf >= threshold:
            words.append({
                "text": word, "confidence": round(conf, 2),
                "left": int(data["left"][i]), "top": int(data["top"][i]),
                "width": int(data["width"][i]), "height": int(data["height"][i])
            })
    return words


def save_outputs(original, gray, binary, words, out_dir, threshold):
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "grayscale.png"), gray)
    cv2.imwrite(str(out_dir / "preprocessed.png"), binary)

    marked = original.copy()
    for word in words:
        x, y, w, h = (word[k] for k in ("left", "top", "width", "height"))
        cv2.rectangle(marked, (x, y), (x+w, y+h), (0, 180, 0), 2)
        cv2.putText(marked, f'{word["confidence"]:.0f}%',
                    (x, max(18, y-5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 120, 0), 1)
    cv2.imwrite(str(out_dir / "annotated_result.png"), marked)

    text = " ".join(w["text"] for w in words)
    (out_dir / "extracted_text.txt").write_text(text + "\n", encoding="utf-8")
    with (out_dir / "recognized_words.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=[
            "text", "confidence", "left", "top", "width", "height"
        ])
        writer.writeheader()
        writer.writerows(words)

    avg = round(sum(w["confidence"] for w in words)/len(words), 2) if words else 0
    report = {
        "confidence_threshold": threshold,
        "accepted_word_count": len(words),
        "average_accepted_confidence": avg,
        "validation_passed": bool(words) and avg >= threshold,
        "extracted_text": text,
        "words": words
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description="OCR invoice/document reader")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--threshold", type=float, default=80.0)
    parser.add_argument("--output", default="output")
    parser.add_argument("--tesseract", default=None,
                        help="Optional path to tesseract.exe")
    args = parser.parse_args()

    if args.tesseract:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract

    original, gray, binary = preprocess(Path(args.image))
    words = recognize(binary, args.threshold)
    report = save_outputs(original, gray, binary, words,
                          Path(args.output), args.threshold)

    print("\\n=== AI INVOICE READER ===")
    print(f"Accepted words (>= {args.threshold}%): {len(words)}")
    print(f"Average accepted confidence: {report['average_accepted_confidence']}%")
    print("Validation:", "PASSED" if report["validation_passed"] else "NOT PASSED")
    print("\\nExtracted text:\\n", report["extracted_text"])
    print("\\nResults saved in:", Path(args.output).resolve())


if __name__ == "__main__":
    main()
