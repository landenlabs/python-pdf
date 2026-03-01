import os
import re
import argparse
import fitz  # pip install pymupdf
import pikepdf  # pip install pikepdf
from pathlib import Path


def classify_pdf(file_path):
    """Analyzes the PDF to determine if it's XFA, Scanned, or Digital."""
    results = {"type": "Unknown", "detail": ""}
    try:
        # 1. Check for XFA
        with pikepdf.Pdf.open(file_path) as pdf:
            # Use .Root (capitalized) or access directly via Root
            if "/AcroForm" in pdf.Root and "/XFA" in pdf.Root.AcroForm:
                results["type"] = "XFA Form"
                return results

        # 2. Check for Scanned vs Digital using PyMuPDF
        doc = fitz.open(file_path)
        text_found = False
        total_image_area = 0.0
        total_page_area = 0.0

        for page in doc:
            page_area = page.rect.width * page.rect.height
            total_page_area += page_area

            if len(page.get_text().strip()) > 10:
                text_found = True

            for block in page.get_text("dict")["blocks"]:
                if block["type"] == 1:  # Image block
                    bbox = block["bbox"]
                    total_image_area += (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        doc.close()

        image_coverage = total_image_area / total_page_area if total_page_area > 0 else 0

        if not text_found and image_coverage > 0.7:
            results["type"] = "Scanned (Image)"
        elif text_found and image_coverage > 0.7:
            results["type"] = "Scanned (OCR'd)"
        else:
            results["type"] = "Native Digital"

    except Exception as e:
        results["type"] = "Error"
        results["detail"] = str(e)

    return results


def main():
    parser = argparse.ArgumentParser(prog="pdf-info", description="Classify PDF types.")
    parser.add_argument("paths", nargs="+", help="One or more file or directory paths.")
    parser.add_argument("--recurse", action="store_true", help="Recurse through directories.")
    parser.add_argument("--exclude-files", action="append", help="Regex to exclude files.")

    args = parser.parse_args()
    exclude_regex = [re.compile(r) for r in args.exclude_files] if args.exclude_files else []

    files_to_process = []

    for p in args.paths:
        path_obj = Path(p)
        if path_obj.is_file():
            files_to_process.append(path_obj)
        elif path_obj.is_dir():
            pattern = "**/*.pdf" if args.recurse else "*.pdf"
            files_to_process.extend(path_obj.glob(pattern))

    for file_path in files_to_process:
        # Check exclusion regex
        if any(r.search(str(file_path)) for r in exclude_regex):
            continue

        result = classify_pdf(file_path)
        print(f"[{result['type']:<15}] {file_path}")
        if result['detail']:
            print(f"    └─ Error: {result['detail']}")


if __name__ == "__main__":
    main()
