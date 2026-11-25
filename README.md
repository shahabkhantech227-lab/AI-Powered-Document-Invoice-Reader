# AI-Powered Document & Invoice Reader

A practical OCR project using OpenCV and Tesseract to extract text from invoice images.

## Features
- Image loading and RGB-to-grayscale conversion
- Gaussian blur, deskewing, and adaptive thresholding
- OCR text extraction with PyTesseract
- 80% word-confidence filter
- Annotated image with bounding boxes
- TXT, CSV, and JSON result files

## Setup (Windows / VS Code)

1. Open this folder in VS Code.
2. Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. Install Python packages:
   ```powershell
   pip install -r requirements.txt
   ```
4. Install the separate Tesseract OCR application for Windows. `pytesseract` is only the Python wrapper.
5. Run:
   ```powershell
   python app.py --image input/sample_invoice.png
   ```
   If Tesseract is not on PATH:
   ```powershell
   python app.py --image input/sample_invoice.png --tesseract "C:\Program Files\Tesseract-OCR\tesseract.exe"
   ```

## Use another image
Place a JPG/PNG in `input/` and run:
```powershell
python app.py --image input/your_invoice.jpg
```

## Generated output
- `output/grayscale.png`
- `output/preprocessed.png`
- `output/annotated_result.png`
- `output/extracted_text.txt`
- `output/recognized_words.csv`
- `output/report.json`

## Project 4 slide alignment
Implements OCR, OpenCV preprocessing, grayscale, Gaussian blur, deskewing, adaptive thresholding, confidence filtering at 80%, and visual confirmation.

**Note:** OCR confidence is an estimate, not a guarantee. Always review extracted invoice details before using them in a business workflow.
