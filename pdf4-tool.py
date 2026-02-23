#!/usr/bin/env python3

import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import os
import re
import sys
import argparse
import csv
import shutil

#
# Windows ?
#    python3 -m pip install pdf2image
#    python3 -m pip install --upgrade pip
#  Installer:
#    python3 -m pip install pyinstaller
#    pyinstaller --onefile --noconsole pdf4-tool.py
#
# Mac
#    brew install poppler
#    brew install tesseract
#    python3 -m pip install opencv-python numpy pdf2image Pillow pytesseract


## Global values
input_name = ""

def parse_arguments():
    parser = argparse.ArgumentParser(description="PDF OCR Checkbox and Label Extractor (--help for full help)",
        epilog="""Example usage:
  pdf4-tool.py --path file100.pdf --region-csv regions1.csv --region-csv regions2.csv
  pdf4-tool.py --path file100.pdf --remove remove.csv --region-csv regions1.csv --region-csv regions2.csv
  pdf4-tool.py --path file100.pdf --remove - --region-csv regions1.csv --region-csv regions2.csv
  
  --switches can be abbreviated, so --remove-csv can be specified as --remove as long as its unique. 

Where region-csv file structure is:
    Bound box defined as upper left x,y and lower right x,y
    Word left or right defining where the text label is relative to the text box
    Text label width in pixels
    Text region name, output with the results
    
    CSV example:
        1247, 450, 1443, 510, "right", 300, "tc"
        668, 871, 776, 1110, "left", 310, "pupil"
        1780, 1622, 1838, 1670, "right", 300, "Pocket mask"
        
Where remove-csv file structure is:
    Removes region of image by filling with white. 
    Use --remove-csv -   to disable default removal.
    Use --save-removal to save a PNG copy of removed PDF image. 
    
    CSV format - Bound boxes as upper left xy, and lower right x, y
    CSV example:   
        0, 550, 1278, 660       # top of image (names)
        0, 1350, 952, 2450      # outline of bodies
        0, 2640, 2556, 3300     # bottom of image
""",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("pdf_path", help="Path to the input PDF image file")
    parser.add_argument("--debug", action="store_true", help="Enable debug image output")
    # parser.add_argument("--list-regions", action="store_true", help="Output list of default CSV regions. ")
    parser.add_argument("--region-csv", action="append", help="Region csv file (see more with --help)")
    parser.add_argument("--remove-csv", action="append", help="Removal csv file (see more with --help)")
    parser.add_argument("--save-removal", action="store_true", help="Save image with regions removed")
    return parser.parse_args()

DEFAULT_REGIONS = [
    # --- Side: right | Label Width: 300 ---
    [1247, 450, 1443, 510, "right", 300, "tc"],
    [1631, 552, 1850, 600, "right", 300, "sex male/female"],
    [110, 871, 173, 1110, "right", 300, "Level of consciousness"],
    [795, 930, 851, 1170, "right", 300, "skin"],
    [1054, 930, 1115, 1170, "right", 300, "skin"],
    [1280, 930, 1336, 1170, "right", 300, "respirations"],
    [1500, 930, 1561, 1180, "right", 300, "respirations"],
    [1732, 930, 1784, 1170, "right", 300, "Circulation - rate"],
    [1732, 930, 1784, 1170, "right", 300, "no label1"],
    [1954, 930, 2010, 1034, "right", 300, "no label2"],
    [2180, 930, 2230, 1034, "right", 300, "no label3"],
    [2180, 930, 2230, 1034, "right", 300, "no label4"],
    [1954, 1120, 2230, 1180, "right", 300, "no label5"],
    [1954, 1120, 2230, 1180, "right", 300, "no label6"],
    [1530, 1380, 1590, 2052, "right", 300, "Emergency care"],
    [1780, 1622, 1838, 1670, "right", 300, "Pocket mask ?"],
    [2120, 1670, 2174, 1775, "right", 300, "no label7"],
    [1960, 1818, 2016, 2054, "right", 300, "no label8"],
    [995, 2090, 1040, 2290, "right", 300, "no label9"],
    [1210, 2090, 1270, 2290, "right", 300, "no label10"],
    [2070, 2248, 2300, 2340, "right", 300, "AED application CRP, Witnessed"],
    [90, 2520, 315, 2590, "right", 300, "Extraction required"],
    [412, 2520, 600, 2590, "right", 300, "DNR"],
    [667, 2520, 891, 2590, "right", 300, "Rough terrain"],
    [1000, 2520, 1410, 2590, "right", 300, "Seat belt"],
    [1518, 2520, 1705, 2590, "right", 300, "helmet"],
    [1774, 2520, 1954, 2590, "right", 300, "airbags"],
    [2177, 2520, 2333, 2590, "right", 300, "compartment intrusion"],

    # --- Side: left | Label Width: 310 ---
    [668, 871, 776, 1110, "left", 310, "pupil"],

    # --- Side: left | Label Width: 500 ---
    [1260, 1420, 1490, 1960, "left", 500, "Assessment"]
]

TEST_REGIONS = [
    [90, 2520, 315, 2590, "right", 300, "Extraction required"],
]

DEFAULT_REMOVAL = [
    [0, 550, 1278, 660],  # top of image (names)
    [0, 1350, 952, 2450],  # outline of bodies
    [0, 2640, 2556, 3300]  # bottom of image
]

pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

def find_poppler_path():
    """Find poppler installation on Mac"""
    possible_paths = [
        '/opt/homebrew/bin',  # M1/M2 Macs
        '/usr/local/bin',  # Intel Macs
        '/opt/local/bin',  # MacPorts
    ]

    for path in possible_paths:
        pdftoppm_path = os.path.join(path, 'pdftoppm')
        if os.path.exists(pdftoppm_path):
            print(f"Found poppler at: {path}", file=sys.stderr)
            return path

    return None


def find_tesseract():
    """Find and configure tesseract"""
    # Try to find tesseract using 'which'
    tesseract_path = shutil.which('tesseract')

    if tesseract_path:
        print(f"Found tesseract at: {tesseract_path}")
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        return True

    # If not found, try common locations
    possible_paths = [
        '/opt/homebrew/bin/tesseract',  # M1/M2 Macs
        '/usr/local/bin/tesseract',  # Intel Macs
        '/opt/local/bin/tesseract',  # MacPorts
    ]

    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found tesseract at: {path}")
            pytesseract.pytesseract.tesseract_cmd = path
            return True

    print("✗ Tesseract not found!")
    print("\nTo install tesseract:")
    print("  brew install tesseract")
    return False


def load_removal_regions(csv_files):
    """Parses CSV files for regions to be whited out, ignoring trailing comments."""
    removal_pts = []
    if not csv_files:
        return removal_pts
    for file_path in csv_files:
        with open(file_path, mode='r') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row: continue
                # Handle potential comments in the last column or row-level string
                # Clean each element to remove trailing comments if present
                clean_row = []
                for item in row:
                    # Split by # and keep only the left side, then strip whitespace
                    clean_item = item.split('#')[0].strip()
                    if clean_item:
                        clean_row.append(clean_item)

                # Syntax: x1, y1, x2, y2
                if len(clean_row) >= 4:
                    removal_pts.append(list(map(int, clean_row[0:4])))
    return removal_pts


def apply_removals(image, removal_regions):
    """Fills specified regions with white (255)."""
    for (x1, y1, x2, y2) in removal_regions:
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), thickness=-1)
    return image


def load_regions(csv_files):
    regions = []
    if not csv_files:
        return regions
    for file_path in csv_files:
        with open(file_path, mode='r') as f:
            reader = csv.reader(f)
            for row in reader:

                if len(row) > 7:
                    # Syntax: x1, y1, x2, y2, side, label_width, label_name
                    x1, y1, x2, y2 = map(int, row[0:4])
                    side = row[4].strip().strip('"')
                    label_w = int(row[5])
                    name = row[6].strip().strip('"')
                    name = re.sub(r'(.+?)#.*', r'\1', name)
                    regions.append([x1, y1, x2, y2, side, label_w, name])
                    ## regions.append({
                    ##     'pts': ((x1, y1), (x2, y2)),
                    ##     'side': side,
                    ##     'label_w': label_w,
                    ##     'name': name
                    ## })
    return regions


def crop_at_first_large_gap_inverse(roi, gap_threshold=15, noise_tolerance=2):
    _, ink_map = cv2.threshold(roi, 127, 1, cv2.THRESH_BINARY_INV)
    projection = np.sum(ink_map, axis=0)
    found_ink = False
    gap_count = 0
    for x, ink_count in enumerate(projection):
        if not found_ink:
            if ink_count > noise_tolerance: found_ink = True
            continue
        if ink_count <= noise_tolerance:
            gap_count += 1
            if gap_count >= gap_threshold:
                return roi[:, 0:max(0, x - gap_count + 1)]
        else:
            gap_count = 0
    return roi


def remove_lines(gray_image):
    """
    Remove horizontal and vertical lines from form
    Input: grayscale numpy array or PIL Image
    Output: grayscale numpy array
    """

    # Convert to numpy array if PIL Image
    if isinstance(gray_image, Image.Image):
        gray = np.array(gray_image)
    else:
        gray = gray_image

    # Ensure it's 2D grayscale
    if len(gray.shape) == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    # Apply threshold
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Remove horizontal lines (min length of 30)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(thresh, [c], -1, (0, 0, 0), 3)

    # Remove vertical lines  (min length of 30)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
    detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(thresh, [c], -1, (0, 0, 0), 3)

    # Invert and Re-binarize to ensure clean binary image (keeps White-on-Black for contour detection)
    result = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return result


def preprocess_pdf(pdf_path, args):
    debug = args.debug

    # Validate PDF file
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    file_size = os.path.getsize(pdf_path)
    print(f"Processing PDF: {pdf_path}  \nFile size: {file_size:,} bytes", file=sys.stderr)

    if file_size == 0:
        raise ValueError("   PDF file is empty")

    # Convert PDF to image
    poppler_path = find_poppler_path()
    images = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)
    original_image = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)

    # Load removal coordinates from CSV
    removal_regions = DEFAULT_REMOVAL
    if args.remove_csv:
        if args.remove_csv == "-":
            removal_regions = []
        else:
            removal_regions = load_removal_regions(args.remove_csv)

    if len(removal_regions) > 0:
        original_image = apply_removals(original_image, removal_regions)
        if args.save_removal:
            cv2.imwrite("debug_" + input_name + "_removal.png", original_image)

    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Clean and Threshold
    denoised = cv2.fastNlMeansDenoising(gray_image, None, 10, 7, 21)
    # Increase contrast
    ## clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ## contrast = clahe.apply(denoised)

    black_white_image = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # Morphological operations to clean up
    ## kernel = np.ones((2, 2), np.uint8)
    ## cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    black_white_image = remove_lines(Image.fromarray(black_white_image))

    if debug:
        cv2.imwrite("debug_" + input_name + "_preprocessed.jpg", black_white_image)
    return black_white_image, original_image


def process_regions(original, binary, regions, debug=False):
    results = []
    # Work on a copy for drawing if debug is enabled
    canvas = original.copy() if debug else original

    for reg in regions:
        x1, y1, x2, y2, side, label_w, name = reg
        pt1 = (x1,y1)
        pt2 = (x2,y2)
        roi = binary[pt1[1]:pt2[1], pt1[0]:pt2[0]]

        # Local search for checked boxes
        cnts, _ = cv2.findContours(cv2.bitwise_not(roi), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            bx, by, bw, bh = cv2.boundingRect(c)
            aspect_ratio = bw / float(bh)
            if 15 < bw < 40 and 30 < bh < 40 and 0.8 < aspect_ratio < 1.3:
                # Check for "ink" in center
                pad = int(bw * 0.3)
                check_roi = roi[by + pad:by + bh - pad, bx + pad:bx + bw - pad]
                # In binary image, 0 is ink (black)
                ink_pct = 1.0 - (cv2.countNonZero(check_roi) / float(check_roi.size)) if check_roi.size > 0 else 0

                if ink_pct > 0.2:
                    # verify checkbox is not text.
                    cb_roi = roi[by:by + bh, bx:bx + bw]
                    text = pytesseract.image_to_string(cb_roi, config='--psm 7').strip()
                    if len(re.sub(r'[^a-zA-Z0-9]', '', text)) > 0:
                        continue

                    # Logic to find label based on CSV side
                    if side == "right":
                        lx, ly = pt1[0] + bx + bw + 5, pt1[1] + by - 5
                    else:  # left
                        lx, ly = pt1[0] + bx - label_w, pt1[1] + by - 5

                    lw, lh = label_w, bh + 10
                    l_roi = binary[ly:ly + lh, lx:lx + lw]
                    l_roi_clean = crop_at_first_large_gap_inverse(l_roi)

                    custom_config = r'--oem 3 --psm 7'  # Single line mode
                    text = pytesseract.image_to_string(l_roi_clean, config=custom_config).strip()
                    clean_text = re.sub(r'[^a-zA-Z0-9]', '', text)
                    # clean_text = re.sub(r'(.+?)[A-Z].*', r'\1', clean_text)  # Tail trim

                    if len(clean_text) == 0:
                        label_text = pytesseract.image_to_string(l_roi, config=custom_config).strip()
                        clean_text = re.sub(r'[^a-zA-Z0-9]', '', label_text)
                        clean_text = re.sub(r'(.+?)[A-Z].*', r'\1',
                                            clean_text)  # Remove trailing appended Capital word,  FaceCc -> Face

                    results.append({
                        'region': name,
                        'label': clean_text,
                        'pos': (pt1[0] + bx, pt1[1] + by) })

                    if debug:
                        cv2.rectangle(canvas, (pt1[0] + bx, pt1[1] + by), (pt1[0] + bx + bw, pt1[1] + by + bh),  (0, 255, 0), 2)
                        lb1 = (lx,ly)
                        lb2 = (lx + lw, ly + lh)
                        cv2.rectangle(canvas, lb1, lb2,  (0, 0, 255), 1)
                        cv2.putText(canvas, clean_text, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                elif debug:
                    cv2.rectangle(canvas, (pt1[0] + bx, pt1[1] + by), (pt1[0] + bx + bw, pt1[1] + by + bh),  (255, 0, 255), 2)

    if debug:
        cv2.imwrite("debug_" + input_name + "_final.png", canvas)
    return results


if __name__ == "__main__":
    args = parse_arguments()

    try:
        base = os.path.basename(args.pdf_path)
        input_name = os.path.splitext(base)[0]

        # Preprocess with removals
        binary_img, color_img = preprocess_pdf(args.pdf_path, args)

        # Load active search regions
        active_regions = load_regions(args.region_csv)
        if not active_regions:
            active_regions = DEFAULT_REGIONS      #  TEST_REGIONS
            print("Using default regions (see --list-regions", file=sys.stderr)


        # Run Detection
        final_results = process_regions(color_img, binary_img, active_regions, args.debug)

        print("Num, Region, Label, Position")
        for i, res in enumerate(final_results, 1):
            print(f"{i:3d}, {res['region']}, {res['label']}, {res['pos']}")

    except Exception as ex:
        print(f"Critical Error: {ex}", file=sys.stderr)