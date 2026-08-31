#!/usr/bin/env python3

import argparse
import sys
import fitz # pip install pymupdf


def extract_text(input_path, output_path=None, first_page=None, last_page=None):
    doc = fitz.open(input_path)

    start = (first_page - 1) if first_page else 0
    end = last_page if last_page else doc.page_count

    text_parts = []
    for page_num in range(start, end):
        page = doc.load_page(page_num)
        text_parts.append(page.get_text())
    doc.close()

    text = "\n".join(text_parts)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved extracted text to: {output_path}")
    else:
        print(text)


def main():
    """
    Parses command-line arguments and extracts text from a PDF file.
    """
    parser = argparse.ArgumentParser(description="Extract text from a PDF file.")
    parser.add_argument("input_file", help="The path to the input PDF file.")
    parser.add_argument("-o", "--output_file",
                        help="The path for the output text file. "
                             "Defaults to printing to stdout.")
    parser.add_argument("--first-page", type=int, default=None,
                        help="First page to extract (1-based). Defaults to the first page.")
    parser.add_argument("--last-page", type=int, default=None,
                        help="Last page to extract (1-based, inclusive). Defaults to the last page.")

    args = parser.parse_args()

    try:
        extract_text(args.input_file, args.output_file, args.first_page, args.last_page)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
