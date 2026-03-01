#!/usr/bin/env python3

import cv2
import numpy as np
from numpy.f2py.auxfuncs import get_kind
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import os
import re
import sys
import argparse
import csv
import shutil
import platform
import traceback
import fitz  # pip install pymupdf
import pikepdf  # pip install pikepdf
from pathlib import Path
import pdfplumber  # python -m pip install pdfplumber
from lxml import etree  # pip install lxml
import xml.etree.ElementTree as ET

#
# Windows
#    Run python only from command line:
#      python -m pip install --upgrade pip
#      python -m pip install opencv-python
#      python -m pip install pdf2image
#      python -m pip install pytesseract
#      python -m pip install lxml
#      python -m pip install pdfplumber
#      pip list
#
#    Run Pycharm
#      .venv\Scripts\activate
#       install all python modules list above
#
#  Build installer on windows:
#    python -m pip install pyinstaller
#    pyinstaller --onefile --console pdf4-tool.py
#
# Mac
#    brew install poppler
#    brew install tesseract
#    python3 -m pip install opencv-python numpy pdf2image Pillow pytesseract


## Global values
input_name = ""
poppler_path = None

def parse_arguments():
    parser = argparse.ArgumentParser(description="PDF OCR Checkbox and Label Extractor (--help for full help)",
        epilog="""Example usage:
    Three modes:
       1. Get information about pdf files
       2. Extract checkboxes from pdf image page #1
       3. Extract checkboxes from pdf XFA page #1
        
    Information:
        pdf4-tool.py --info file100.pdf file200.pdf 
        pdf4-tool.py --info --recurse dir1 dir2/subdir  file1.pdf
        pdf4-tool.py --info --exclude-files "xfa.*pdf" dir1 dir2 
        
    Extract checkboxes from pdf image on page #1
        Checkbox extract performs the following steps:
        1. Converts image to black and white and removes long vertical and horizontal lines.
        2. Removes regions (default is top where name info is, diagram of body and bottom 
        3. Using default or provided regions, locate checkboxes and labels (right or left)
        
        Use default checkbox regions and default removal regions on page #1
           pdf4-tool.py --extract  file100.pdf
        Use default regions and save various phases of process as jpg or png images
           pdf4-tool.py --extract --debug file100.pdf
        User provided regions:
           pdf4-tool.py --extract --region-csv regions1.csv --region-csv regions2.csv  file100.pdf
           pdf4-tool.py --extract --remove remove.csv --region-csv regions1.csv --region-csv regions2.csv  file100.pdf 
 
    Extract or dump information from XFA (or normal pdf) files:
        pdf4-tool.py --extract  pdf_xfa.pdf
        pdf4-tool.py --extract --verbose pdf_xfa.pdf    
        
           
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
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("paths", nargs="+", help="Extract: pdf file, Info: files or directories")

    group = parser.add_mutually_exclusive_group(required=True)
    # Add arguments to that specific group object
    group.add_argument('--info', action='store_true', help="Get information about pdf file(s)")
    group.add_argument('--extract', action='store_true', help="Extract data from pdf file")

    extract_group = parser.add_argument_group("Extraction Specifics")
    extract_group.add_argument("--debug", action="store_true", help="Enable debug image output")
    parser.add_argument("--list-regions", action="store_true", help="Output list of default CSV regions. ")
    extract_group.add_argument("--region-csv", action="append", help="Region csv file (see more with --help)")
    extract_group.add_argument("--remove-csv", action="append", help="Removal csv file (see more with --help)")
    extract_group.add_argument("--save-removal", action="store_true", help="Save image with regions removed")

    info_group = parser.add_argument_group("Info Specifics")
    info_group.add_argument("--recurse", action="store_true", help="Recurse through directories.")
    info_group.add_argument("--exclude-files", action="append", help="Regex to exclude files.")
    return parser.parse_args()

DEFAULT_REGIONS = [
    # --- Side: right | Label Width: 300 ---
    [1247, 450, 1443, 510, "right", 300, "tc"],
    [1631, 552, 1850, 600, "right", 300, "sex male/female"],
    [110, 871, 173, 1110, "right", 300, "Lvl_consc"],
    [795, 930, 851, 1170, "right", 300, "skin_color"],
    [1054, 930, 1115, 1170, "right", 300, "skin_temp"],
    [1280, 930, 1336, 1170, "right", 300, "resp_rate"],
    [1500, 930, 1561, 1180, "right", 300, "resp_rhyt"],
    [1732, 930, 1784, 1170, "right", 300, "Circu_rate"],
    [1954, 930, 2010, 1034, "right", 300, "Circu_rhytm"],
    [2180, 930, 2230, 1034, "right", 300, "Circu_stren"],
    [1954, 1120, 2230, 1180, "right", 300, "Circu_capil"],
    [1530, 1380, 1590, 2052, "right", 300, "Emerg_care1"],
    [1780, 1622, 1838, 1670, "right", 300, "Pocket mask ?"],
    [2120, 1670, 2174, 1775, "right", 300, "Emerg_care2"],
    [1960, 1818, 2016, 2054, "right", 300, "Emerg_care3"],
    [995, 2090, 1040, 2290, "right", 300, "Med_hist1"],
    [1210, 2090, 1270, 2290, "right", 300, "Med_hist2"],
    [2070, 2248, 2300, 2290, "right", 300, "CPR"],
    [2070, 2300, 2300, 2340, "right", 300, "Witnessed"],
    [90, 2520, 315, 2590, "right", 300, "Extr-reqd"],
    [412, 2520, 600, 2590, "right", 300, "DNR"],
    [667, 2520, 891, 2590, "right", 300, "Rough-terrain"],
    [1000, 2520, 1410, 2590, "right", 300, "Seat_belt"],
    [1518, 2520, 1705, 2590, "right", 300, "helmet"],
    [1774, 2520, 1954, 2590, "right", 300, "airbags"],
    [2177, 2520, 2333, 2590, "right", 300, "comp_intr"],

    # --- Side: left | Label Width: 310 ---
    [668, 871, 776, 1110, "left", 310, "pupil"],

    # --- Side: left | Label Width: 500 ---
    [1260, 1420, 1390, 1960, "left", 500, "Asses_WNI"],
    [1400, 1420, 1490, 1960, "left", 500, "Asses_ABN"]
]

TEST_REGIONS = [
    [90, 2520, 315, 2590, "right", 300, "Extraction required"],
]

DEFAULT_REMOVAL = [
    [0, 550, 1278, 660],  # top of image (names)
    [0, 1350, 952, 2450],  # outline of bodies
    [0, 2640, 2556, 3300]  # bottom of image
]



def is_windows():
   return platform.system() == "Windows"

def find_poppler_path():
    """Find poppler installation on Mac"""
    global poppler_path

    # 1. Return immediately if already cached
    if poppler_path is not None:
        return poppler_path

    look_for = 'pdftoppm.exe' if is_windows() else 'pdftoppm'

    # 2. Check System PATH first (The "Default" Python/System location)
    system_match = shutil.which(look_for)
    if system_match:
        # shutil.which returns the full path to the exe; we need the directory
        poppler_path = os.path.dirname(system_match)
        return poppler_path

    # 3. Search specific known locations
    possible_paths = [
        '/opt/homebrew/bin',  # M1/M2 Macs
        '/usr/local/bin',  # Intel Macs
        '/opt/local/bin',  # MacPorts
        'c:/opt/pdf/Poppler-pdf2img-25.12.0-0/poppler-25.12.0/Library/bin',
    ]

    for path in possible_paths:
        pdftoppm_path = os.path.join(path, look_for)
        if os.path.exists(pdftoppm_path):
            # print(f"Found {look_for} at: {path}", file=sys.stderr)
            poppler_path = path
            return poppler_path


    print(f"FAILED to find  {look_for}", file=sys.stderr)
    return poppler_path


def set_tesseract_path(args):
    """Only configure tesseract if it isn't already working."""

    # 1. Quick Check: Is it already working?
    try:
        # If this returns a version, Tesseract is already in the system PATH.
        # We don't need to do any manual searching.
        pytesseract.get_tesseract_version()
        if args.verbose:
            print("Tesseract already functional via system PATH.", file=sys.stderr)
        return True
    except (pytesseract.TesseractNotFoundError, EnvironmentError):
        if args.verbose:
            print("Tesseract not found in PATH, searching common locations...", file=sys.stderr)

    # 2. If not found, look for the executable
    look_for = 'tesseract.exe' if is_windows() else 'tesseract'

    # Check 'which' (shutil finds it if it's in a non-standard PATH)
    tesseract_path = shutil.which(look_for)
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        return True

    # 3. Last Resort: Hardcoded common locations
    possible_paths = [
        '/opt/homebrew/bin/tesseract',
        '/usr/local/bin/tesseract',
        '/opt/local/bin/tesseract',
        'c:/opt/pdf/Tesseract-OCR/tesseract.exe',  # Added exe to the specific path
    ]

    for path in possible_paths:
        # Use the path directly if it points to the file, or join if it's a dir
        target = path if path.endswith(look_for) else os.path.join(path, look_for)

        if os.path.exists(target):
            pytesseract.pytesseract.tesseract_cmd = target
            return True
        elif args.verbose:
            print(f" {look_for} not in {path}", file=sys.stderr)

    print(f"ERROR: {look_for} not found!", file=sys.stderr)
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
    print(f"Processing PDF: {pdf_path}  \nFile size= {file_size:,} bytes", file=sys.stderr)

    if file_size == 0:
        raise ValueError("   PDF file is empty")

    # Convert PDF to image
    poppler_path = find_poppler_path()
    images = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)
    image_cnt = len(images)
    print(f"Image pages found={image_cnt}", file=sys.stderr)
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
    canvas1 = original.copy() if debug else original
    canvas2 = original.copy() if debug else original

    reg_idx = 0
    for reg in regions:
        x1, y1, x2, y2, side, label_w, name = reg
        pt1 = (x1,y1)
        pt2 = (x2,y2)
        roi = binary[pt1[1]:pt2[1], pt1[0]:pt2[0]]

        if debug:
            cv2.rectangle(canvas1, pt1, pt2, (0, 0, 255), 2)
            reg_idx += 1
            cv2.putText(canvas1, f"{name}", (x1-10, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

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
                        cv2.rectangle(canvas2, (pt1[0] + bx, pt1[1] + by), (pt1[0] + bx + bw, pt1[1] + by + bh),  (0, 255, 0), 2)
                        lb1 = (lx,ly)
                        lb2 = (lx + lw, ly + lh)
                        cv2.rectangle(canvas2, lb1, lb2,  (0, 0, 255), 1)
                        cv2.putText(canvas2, clean_text, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                elif debug:
                    cv2.rectangle(canvas2, (pt1[0] + bx, pt1[1] + by), (pt1[0] + bx + bw, pt1[1] + by + bh),  (255, 0, 255), 2)

    if debug:
        cv2.imwrite("debug_" + input_name + "_region.png", canvas1)
        cv2.imwrite("debug_" + input_name + "_final.png", canvas2)
    return results

def pdf_extract_from_images(args):

    set_tesseract_path(args)
    pdf_path = args.paths[0]
    base = os.path.basename(pdf_path)
    input_name = os.path.splitext(base)[0]

    # Preprocess with removals
    binary_img, color_img = preprocess_pdf(pdf_path, args)

    # Load active search regions
    active_regions = load_regions(args.region_csv)
    if not active_regions:
        active_regions = DEFAULT_REGIONS  # TEST_REGIONS
        print("Using default regions, see --list-regions", file=sys.stderr)

    # Run Detection
    final_results = process_regions(color_img, binary_img, active_regions, args.debug)

    print("Num, Region, Label, Position")
    for i, res in enumerate(final_results, 1):
        print(f"{i:3d}, {res['region']}, {res['label']}, {res['pos']}")

## ---------------------------------------------------------------------------------------------------------------------

def pdf_extract_from_xfa(args):
    pdf_path = args.paths[0]

    if args.verbose:
        dump_all_xfa_parts(pdf_path)
        return

    with pikepdf.open(pdf_path) as pdf:
        try:
            xfa = pdf.Root.AcroForm.XFA
        except (AttributeError, KeyError):
            print("No XFA data found.")
            return

        # Locate the datasets packet
        xml_data = b""
        # XFA is an array of [key, stream, key, stream...]
        for i in range(0, len(xfa), 2):
            if str(xfa[i]) == "datasets":
                xml_data = xfa[i + 1].read_bytes()
                break

        if not xml_data:
            print("No dataset stream found.")
            return

        # Parse XML
        root = etree.fromstring(xml_data)

        print(f"{'Field (XML Tag)':<45} | {'Content/Value'}")
        print("-" * 70)

        # We iterate through everything in the 'data' portion of the XML
        # Usually, the data is under a <xfa:data> or <topmostSubform> tag
        for element in root.iter():
            # Get the clean tag name (no namespaces)
            name = element.tag.split('}')[-1]

            # Extract text if it exists
            value = element.text.strip() if element.text else None

            # Logic: If it has text and no children, it's a standard field.
            # If it has text AND children, it might be a formatted text block.
            if value != 0:
                print(f"{name}")


def dump_all_xfa_parts(pdf_path):
    print(f"--- XFA Dump for: {pdf_path} ---")

    with pikepdf.open(pdf_path) as pdf:
        try:
            # Check for AcroForm and XFA existence
            if "/AcroForm" not in pdf.Root or "/XFA" not in pdf.Root.AcroForm:
                print("No XFA content found.")
                return

            xfa = pdf.Root.AcroForm.XFA
        except (AttributeError, KeyError):
            print("Error accessing XFA structure.")
            return

        # XFA is an array of [name, stream, name, stream...]
        for i in range(0, len(xfa), 2):
            part_name = str(xfa[i])
            raw_bytes = xfa[i + 1].read_bytes()

            print(f"\n>> Part: {part_name}")
            print("-" * 30)

            try:
                # Parse XML from bytes
                root = ET.fromstring(raw_bytes)

                # Iterate through all elements with a text value
                for elem in root.iter():
                    # Clean up the tag name (remove XML namespaces if present)
                    tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag

                    # Only print if there is actual text content (not just whitespace)
                    if elem.text and elem.text.strip():
                        value = elem.text.strip()
                        print(f"{tag:30} : {value}")

            except ET.ParseError:
                print(f"[Binary or Malformed XML content: {len(raw_bytes)} bytes]")


## ---------------------------------------------------------------------------------------------------------------------
def show_pdf_info(file_path):
    """Analyzes the PDF to determine if it's XFA, Scanned, or Digital."""
    results = {"type": "Unknown", "pages": "-", "images": "-"}
    try:
        # 1. Check for XFA
        with pikepdf.Pdf.open(file_path) as pdf:
            # Use .Root (capitalized) or access directly via Root
            if "/AcroForm" in pdf.Root and "/XFA" in pdf.Root.AcroForm:
                results["type"] = "XFA Form"
                results["pages"] = "-"
                results["images"] = "-"
                return results

        # 2. Check for Scanned vs Digital using PyMuPDF
        doc = fitz.open(file_path)
        text_found = False
        total_image_area = 0.0
        total_page_area = 0.0

        page_cnt = len(doc)
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


        type = ""
        if not text_found and image_coverage > 0.7:
            type = "Scanned (Image)"
        elif text_found and image_coverage > 0.7:
            type = "Scanned (OCR'd)"
        else:
            type = "Native Digital"

        image_cnt = 0
        try:
            poppler_path = find_poppler_path()
            images = convert_from_path(file_path, dpi=300, poppler_path=poppler_path)
            image_cnt = len(images)
            # print(f"Image pages found={image_cnt}", file=sys.stderr)
        except Exception as ex:
            # print(f"Error: {ex}", file=sys.stderr)
            image_cnt = 0

        results["type"] = type
        results["pages"] = f"{page_cnt}"
        results["images"] = f"{image_cnt}"

    except Exception as e:
        results["type"] = str(e)
        results["pages"] = "-"
        results["images"] = "-"

    return results


def pdf_info(args):
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

        result = show_pdf_info(file_path)
        print(f"[{result['type']:<15}, Pages:{result['pages']:<5}, Images:{result['images']:<5}] {file_path}  ")


if __name__ == "__main__":

    try:
        # Get the name of the file (e.g., 'pdf-info.exe' or 'pdf-extract.py')
        program_name = Path(sys.argv[0]).stem.lower()
        args = parse_arguments()
        if args.info:
            pdf_info(args)
        elif args.extract:
            if args.list_regions:
                print("\n\nDefault extract regions:")
                for region in DEFAULT_REGIONS:
                    print(", ".join(map(str, region)))

                print("\n\nDefault removal regions:")
                for region in DEFAULT_REMOVAL:
                    print(", ".join(map(str, region)))
                sys.exit(0)

            result = show_pdf_info(args.paths[0])
            if result["images"] != "-":
                pdf_extract_from_images(args)
            else:
                pdf_extract_from_xfa(args)

        else:
            print(f"Must specify --info or --extract.", file=sys.stderr)

    except Exception as ex:
        # This captures the full trace, including file and line number
        error_details = traceback.format_exc()
        print(f"Critical Error:\n{error_details}", file=sys.stderr)
        print("[Done]", file=sys.stderr)