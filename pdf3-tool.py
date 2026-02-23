#!/usr/bin/env python3

import argparse
import os
import fitz # pip install pymupdf


def pymupdf_xfa_pdf(input_path, output_path):
    doc = fitz.open(input_path)
    if doc.is_xfa:
        # To handle XFA forms, you need to save with garbage collection
        # This will expand the XFA data into a regular PDF
        doc.save(output_path, garbage=4, clean=True, deflate=True)
    else:
        # For non-XFA PDFs, a simple save is enough
        doc.save(output_path)
    doc.close()

    print(f"Saved flattened PDF to: {output_path}")
    print("Conversion complete.")


def main():
    """
    Parses command-line arguments and initiates the PDF flattening process.
    """
    parser = argparse.ArgumentParser(description="Flatten a dynamic XFA PDF to a static PDF.")
    parser.add_argument("input_file", help="The path to the input XFA PDF file.")
    parser.add_argument("-o", "--output_file",
                        help="The path for the output static PDF file. "
                             "Defaults to 'static_<input_filename>' in the same directory.")

    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file

    if not output_file:
        # Create a default output name if one isn't provided
        base_name = os.path.basename(input_file)
        dir_name = os.path.dirname(input_file)
        if dir_name:
            output_file = os.path.join(dir_name, f"static_{base_name}")
        else:
            output_file = f"static_{base_name}"

    pymupdf_xfa_pdf(input_file, output_file)

if __name__ == "__main__":
    main()
