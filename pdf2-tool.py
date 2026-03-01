#!/usr/bin/env python3

import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import os
import shutil
from collections import Counter, defaultdict
import sys
import re

#  python3 -m pip install --upgrade pip
# Windows ?
#    pip install pdf2image
#    python3 -m pip install pdf2image
# Mac
#    brew install poppler
#    brew install tesseract


# Example Usage with your provided corners:
search_regions_label_right = [
    ((1247, 450), (1443, 510)),  # tc
    ((1631, 552), (1850, 600)),  # sex male/female
    ((110, 871), (173, 1110)),  # Level of consciousness
    #    ((668, 871), (776, 1110)),  # pupil  (labels on left)
    ((795, 930), (851, 1170)),  # skin
    ((1054, 930), (1115, 1170)),  # skin
    ((1280, 930), (1336, 1170)),  # respirations
    ((1500, 930), (1561, 1180)),  # respirations
    ((1732, 930), (1784, 1170)),  # Circulation - rate
    ((1732, 930), (1784, 1170)),
    ((1954, 930), (2010, 1034)),
    ((2180, 930), (2230, 1034)),
    ((2180, 930), (2230, 1034)),
    ((1954, 1120), (2230, 1180)),
    ((1954, 1120), (2230, 1180)),
    #    ((1260, 1420), (1490, 1960)),   # Assessment (labels on left)
    ((1530, 1380), (1590, 2052)),  # Emergency care

    ((1780, 1622), (1838, 1670)),  # Pocket mask ?
    ((2120, 1670), (2174, 1775)),
    ((1960, 1818), (2016, 2054)),

    ((995, 2090), (1040, 2290)),
    ((1210, 2090), (1270, 2290)),

    ((2070, 2248), (2300, 2340)),  # AED application CRP, Witnessed

    ((90, 2520), (315, 2590)),  # Extraction required
    ((412, 2520), (600, 2590)),  # DNR
    ((667, 2520), (891, 2590)),  # Rough terrain
    ((1000, 2520), (1410, 2590)),  # Seat belt
    ((1518, 2520), (1705, 2590)),  # helmet
    ((1774, 2520), (1954, 2590)),  # airbags
    ((2177, 2520), (2333, 2590)),  # compartment intrusion
]

search_regions_label_left1 = [
    ((668, 871), (776, 1110)),  # pupil  (labels on left)

]
search_regions_label_left2 = [
    ((1260, 1420), (1490, 1960)),  # Assessment (labels on left)
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
            print(f"Found poppler at: {path}")
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

    debug_file = f"debug_gray1.jpg"
    cv2.imwrite(debug_file, gray)
    print(f" Debug image saved as {debug_file} ---", file=sys.stderr)

    # Apply threshold
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Remove horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(thresh, [c], -1, (0, 0, 0), 3)

    # Remove vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
    detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(thresh, [c], -1, (0, 0, 0), 3)

    # Invert and Re-binarize to ensure clean binary image (keeps White-on-Black for contour detection)
    result = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    debug_file = f"debug_no_lines.jpg"
    cv2.imwrite(debug_file, result)
    print(f" Debug image saved as {debug_file} ---", file=sys.stderr)
    return result


def preprocess_for_checkbox_detection(pdf_path, page_num=0):
    """Complete preprocessing pipeline"""

    # Validate PDF file
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    file_size = os.path.getsize(pdf_path)
    print(f"Processing PDF: {pdf_path}")
    print(f"   File size: {file_size:,} bytes")

    if file_size == 0:
        raise ValueError("   PDF file is empty")

    # Convert PDF to image
    poppler_path = find_poppler_path()
    try:
        if poppler_path:
            print(f"   Using poppler from: {poppler_path}")
            images = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)
        else:
            print("   Trying default poppler location...")
            images = convert_from_path(pdf_path, dpi=300)
    except Exception as e:
        print(f"\n✗ ERROR Reading PDF, type: {type(e).__name__} error: {e}")
        raise

    if not images:
        raise ValueError("PDF conversion returned no images")

    print(f"✓ Successfully read {len(images)} page(s)")

    if page_num >= len(images):
        raise ValueError(f"Page {page_num} does not exist (PDF has {len(images)} pages)")

    image = images[page_num]
    print(f"Processing page {page_num}, size: {image.size}")

    # Convert to OpenCV format
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    debug_file = f"debug_grey.jpg"
    cv2.imwrite(debug_file, gray)
    print(f" Debug image saved as {debug_file} ---", file=sys.stderr)

    # 1. Denoise
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    debug_file = f"debug_denoised.jpg"
    cv2.imwrite(debug_file, denoised)
    print(f" Debug image saved as {debug_file} ---", file=sys.stderr)

    # 2. Increase contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(denoised)
    debug_file = f"debug_contrast.jpg"
    cv2.imwrite(debug_file, contrast)
    print(f" Debug image saved as {debug_file} ---", file=sys.stderr)

    # 3. Deskew if needed
    if True:
        coords = np.column_stack(np.where(contrast > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) > 0.5:
            (h, w) = contrast.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            contrast = cv2.warpAffine(contrast, M, (w, h),
                                      flags=cv2.INTER_CUBIC,
                                      borderMode=cv2.BORDER_REPLICATE)
            print(f"Skew image by {angle} degrees")

    # 4. Adaptive threshold (better than simple threshold)
    binary = cv2.adaptiveThreshold(contrast, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # 5. Morphological operations to clean up
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    debug_file = f"debug_cleaned.jpg"
    cv2.imwrite(debug_file, cleaned)
    print(f" Debug image saved as {debug_file} ---", file=sys.stderr)

    height = cleaned.shape[0]
    width = cleaned.shape[1]
    channels = img.shape[2] if len(img.shape) > 2 else 1
    print(f"Width: {width}, Height: {height}, Channels: {channels}")
    #  Width: 2556, Height: 3300, Channels: 3

    # Define your two corners (x, y)
    fillcolor = (255, 255, 255)
    x = 100;
    #  Remove outline of body.
    pt1 = (x, int(height/2-300))  # Top-left corner
    pt2 = (int(x + width/3), int(height/2+800))  # Bottom-right corner
    # Use thickness = -1 to fill the shape
    cv2.rectangle(cleaned, pt1, pt2, fillcolor, thickness=-1)

    # Bottom of image.
    pt1 = (0, int(height - height/5))
    pt2 = (width,  height)
    cv2.rectangle(cleaned, pt1, pt2, fillcolor, thickness=-1)

    # Top name field
    pt1 = (0, int( height / 6))
    pt2 = (int(width / 2), int(height / 5))
    cv2.rectangle(cleaned, pt1, pt2, fillcolor, thickness=-1)

    debug_file = f"debug_filled.jpg"
    cv2.imwrite(debug_file, cleaned)
    print(f" Debug image saved as {debug_file} ---", file=sys.stderr)

    return cleaned, img  # Both are numpy arrays

def detect_all_checkboxes(image):
    """Detect checkbox regions"""
    # Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    checkboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Filter by size (adjust based on your forms)
        if 15 < w < 40 and 15 < h < 40:
            aspect_ratio = w / float(h)

            # Check if square-ish
            if 0.8 < aspect_ratio < 1.2:
                # Calculate fill percentage
                roi = image[y:y + h, x:x + w]
                fill_percentage = cv2.countNonZero(roi) / (w * h)

                # Determine if checked (adjust threshold as needed)
                is_checked = fill_percentage > 0.3

                if is_checked:
                    checkboxes.append({
                        'x': x, 'y': y, 'w': w, 'h': h,
                        'checked': is_checked,
                        'fill_percentage': fill_percentage
                    })

    print(f"Found {len(checkboxes)} checkbox(es)")
    return checkboxes


def improved_checkbox_detection2(image):
    # 1. Use CCOMP to get hierarchy (External vs Internal)
    # hierarchy: [Next, Previous, First_Child, Parent]
    contours, hierarchy = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is None: return []

    potential_boxes = []
    rows = defaultdict(list)

    for i, cnt in enumerate(contours):
        # Ignore internal contours (checkmarks/holes) by checking if they have a parent
        if hierarchy[0][i][3] != -1:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)

        # Basic size and squareness filter
        if 15 < w < 50 and 15 < h < 50 and 0.8 < aspect_ratio < 1.2:
            box_data = {'x': x, 'y': y, 'w': w, 'h': h, 'cnt': cnt}
            potential_boxes.append(box_data)

            # Group by Y-coordinate (rounded to ignore slight skews)
            row_key = round(y / 10) * 10
            rows[row_key].append(box_data)

    # 3. Filter by "Alignment"
    # Only keep boxes that have at least one neighbor on the same horizontal plane
    final_checkboxes = []
    for row_y, boxes in rows.items():
        if len(boxes) > 1:  # Adjust to > 0 if you expect single-box rows
            final_checkboxes.extend(boxes)
        else:
            # Optional: Check if the single box is exceptionally 'box-like'
            # (e.g., using cv2.approxPolyDP to ensure 4 corners)
            peri = cv2.arcLength(boxes[0]['cnt'], True)
            approx = cv2.approxPolyDP(boxes[0]['cnt'], 0.04 * peri, True)
            if len(approx) == 4:
                final_checkboxes.extend(boxes)

    print(f"OLD - Found {len(final_checkboxes)} checkbox(es)")
    return final_checkboxes


def improved_checkbox_detection(image):
    # 1. Use CCOMP to get hierarchy
    contours, hierarchy = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None: return []

    potential_boxes = []
    for i, cnt in enumerate(contours):
        # Ignore internal contours (holes/checkmarks)
        ## if hierarchy[0][i][3] != -1:
        ##    continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)

        # Basic size and squareness filter
        if 15 < w < 40 and 15 < h < 40 and 0.8 < aspect_ratio < 1.2:
            if hierarchy[0][i][3] != -1:  # internal contours (holes/checkmarks)
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                # 4 is perfect square. 5 or 6 usually accounts for a "noisy" corner in a scan.
                # if 4 <= len(approx) <= 6:
                if len(approx) == 4:
                    potential_boxes.append({'x': x, 'y': y, 'w': w, 'h': h, 'cnt': cnt})
            else:
                potential_boxes.append({'x': x, 'y': y, 'w': w, 'h': h, 'cnt': cnt})

    # 2. Sort boxes by Y coordinate
    # This allows us to only check immediate neighbors in the list
    potential_boxes.sort(key=lambda b: b['y'])

    # 3. Filter by "Alignment" using a proximity check
    final_checkboxes = []
    y_tolerance = 10

    num_boxes = len(potential_boxes)
    for i in range(num_boxes):
        current = potential_boxes[i]
        has_neighbor = False

        # Check boxes AFTER this one in the sorted list
        for j in range(i + 1, num_boxes):
            neighbor = potential_boxes[j]
            # If we move beyond the tolerance, we can stop looking further down
            if (neighbor['y'] - current['y']) > y_tolerance:
                break
            has_neighbor = True
            break  # Found at least one neighbor

        # Check boxes BEFORE this one in the sorted list
        if not has_neighbor:
            for j in range(i - 1, -1, -1):
                neighbor = potential_boxes[j]
                if (current['y'] - neighbor['y']) > y_tolerance:
                    break
                has_neighbor = True
                break

        # Decision Logic
        if has_neighbor:
            final_checkboxes.append(current)
        else:
            # Fallback: If isolated, strictly verify it's a 4-cornered polygon
            peri = cv2.arcLength(current['cnt'], True)
            approx = cv2.approxPolyDP(current['cnt'], 0.04 * peri, True)
            if len(approx) == 4:
                final_checkboxes.append(current)

    print(f"Found {len(final_checkboxes)} checkbox(es)")
    return final_checkboxes


def find_checked_boxes_in_regions(original, image, regions, debug_draw):
    """
    image: The BGR or Grayscale image
    regions: List of tuples [((x1, y1), (x2, y2)), ...]
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    found_checkboxes = []

    for (pt1, pt2) in regions:
        # 1. Extract the Region of Interest (ROI)
        # Using [y1:y2, x1:x2]
        x1, y1 = pt1
        x2, y2 = pt2
        roi = gray[y1:y2, x1:x2]
        cv2.rectangle(original, pt1, pt2, (255, 0, 0), thickness=1)

        # 2. Local Aggressive Pre-processing
        # Apply Gaussian Blur to reduce noise from scanning/printing
        blurred = cv2.GaussianBlur(roi, (3, 3), 0)

        # Adaptive thresholding to handle local shadows/ink density
        # This makes the ink "pop" against the paper
        thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # 3. Find contours within this specific region
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            bx, by, bw, bh = cv2.boundingRect(c)
            aspect_ratio = bw / float(bh)

            # 4. Filter for checkbox-like shapes
            # Aggressive sizing: looking for boxes between 12 and 60 pixels
            if 15 < bw < 40 and 30 < bh < 40 and 0.8 < aspect_ratio < 1.3:

                # Verify it's not a tiny speck using area
                if cv2.contourArea(c) < 100:
                    continue

                # 5. Check if it's "Checked"
                # We look at the inner 70% of the box to avoid border interference
                inner_pad_w = int(bw * 0.3)
                inner_pad_h = int(bh * 0.3)
                check_roi = thresh[by + inner_pad_h: by + bh - inner_pad_h, bx + inner_pad_w: bx + bw - inner_pad_w]

                if check_roi.size > 0:
                    # Image is inverted
                    white_cnt = cv2.countNonZero(check_roi)
                    white_pct = white_cnt / float(check_roi.size)
                    is_checked = 0.1 < white_pct < 0.5

                    # If the center has significant ink (e.g. > 15%), it's checked
                    if is_checked:
                        found_checkboxes.append({
                            'x': x1 + bx,
                            'y': y1 + by,
                            'w': bw,
                            'h': bh,
                            'checked': is_checked,
                            'fill_percentage': white_pct
                        })

                    if debug_draw:
                        pt1a = (x1 + bx-1, y1 + by-1)
                        pt2a = (x1 + bx+2 + bw, y1 + by + bh+2)
                        if is_checked:
                            cv2.rectangle(original, pt1a, pt2a, (0, 255, 0), thickness=2)
                        else:
                            cv2.rectangle(original, pt1a, pt2a, (0, 0, 255), thickness=2)

    print(f"Region found {len(found_checkboxes)} checkbox(es)")
    return found_checkboxes

def detect_checkboxes_adaptive(original, image):
    ischecked = []
    unchecked = []

    """Detect checkbox regions and categorize by size consistency"""
    boxes = improved_checkbox_detection(image)
    for box in boxes:
        x, y, w, h = box['x'], box['y'], box['w'], box['h']

        pad = int(w * 0.25)
        inner_x = x + pad
        inner_y = y + pad
        inner_w = w - (pad * 2)
        inner_h = h - (pad * 2);

        # Ensure dimensions are valid
        if inner_w <= 0 or inner_h <= 0: continue

        roi = image[inner_y:inner_y + inner_h, inner_x:inner_x + inner_w]
        white_cnt = cv2.countNonZero(roi)
        black_pct = 1 - white_cnt / (inner_w * inner_h)

        # Determine if checked (adjust threshold as needed)
        is_checked = black_pct > 0.2

        pt1 = (inner_x, inner_y)
        pt2 = (inner_x + inner_w, inner_y + inner_h)

        if is_checked:
            ischecked.append({
                'x': x, 'y': y, 'w': w, 'h': h,
                'checked': is_checked,
                'fill_percentage': black_pct
            })
            # cv2.rectangle(original, pt1, pt2, (0, 255, 0), thickness=2)
        else:
            unchecked.append({
                'x': x, 'y': y, 'w': w, 'h': h,
                'checked': is_checked,
                'fill_percentage': black_pct
            })
            # cv2.rectangle(original, pt1, pt2, (0, 0, 255), thickness=2)

    # debug_file = f"debug_boxes.jpg"
    # cv2.imwrite(debug_file, original)
    # print(f" Debug image saved as {debug_file} ---", file=sys.stderr)

    print(f"Checked boxes - Standard: {len(ischecked)} | Unusual: {len(unchecked)}")
    return ischecked, unchecked

def detect_checkboxes_by_region(original, image, zone ):

    debug_draw = True
    if debug_draw:
        original = original.copy()

    boxes = []
    if zone == 0:
        boxes = find_checked_boxes_in_regions(original, image, search_regions_label_right, debug_draw)
        debug_file = f"debug_box_label_right.jpg"
    elif zone == 1:
        boxes = find_checked_boxes_in_regions(original, image, search_regions_label_left1, debug_draw)
        debug_file = f"debug_box_label_left1.jpg"
    else:
        boxes = find_checked_boxes_in_regions(original, image, search_regions_label_left2, debug_draw)
        debug_file = f"debug_box_label_left2.jpg"

    cv2.imwrite(debug_file, original)
    print(f" Debug image saved as {debug_file} ---", file=sys.stderr)

    print(f"Checked boxes by region  {len(boxes)}  ")
    return boxes

def detect_checkboxes_adaptive2(original, image):
    all_candidates = []

    """Detect checkbox regions and categorize by size consistency"""
    contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # # Phase 1: Initial Filter (Broad range to catch all potential boxes)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Broad square-ish filter
        if 10 < w < 30 and 10 < h < 30:
            aspect_ratio = w / float(h)
            if 0.7 < aspect_ratio < 1.3:
                pad = int(w / 3)
                inner_x = x + pad
                inner_y = y + pad
                inner_w = w - (pad * 2)
                inner_h = h - (pad * 2);
                roi = image[inner_y:inner_y + inner_h, inner_x:inner_x + inner_w]
                fill_pct = cv2.countNonZero(roi) / (inner_w * inner_h)

                # Determine if checked (adjust threshold as needed)
                is_checked = fill_pct > 0.2

                pt1 = (inner_x, inner_y)
                pt2 = (inner_x + inner_w, inner_y + inner_h)
                # cv2.rectangle(original, pt1, pt2, (0, 255, 0), thickness=1)

                # if is_checked:
                if True:
                    all_candidates.append({
                        'x': x, 'y': y, 'w': w, 'h': h,
                        'checked': is_checked,
                        'fill_percentage': fill_pct
                    })

    if not all_candidates:
        return [], []

    # Phase 2: Identify the "Standard" Size
    # We round to the nearest 2 pixels to account for slight scanning noise
    ## size_counts = Counter([(round(c['w'] / 2) * 2, round(c['h'] / 2) * 2) for c in all_candidates])
    size_counts = Counter([(c['w'] , c['h']) for c in all_candidates])
    most_common_size, _ = size_counts.most_common(1)[0]
    std_w, std_h = most_common_size
    print(f"Common box width {std_w} height {std_h}")

    # Phase 3: Split into two lists
    standard_checkboxes = []
    unusual_checkboxes = []

    # Tolerance for being "similar" (e.g., +/- 3 pixels)
    tolerance = 4
    margin = 2

    for box in all_candidates:
        # is_width_sim = abs(box['w'] - std_w) <= tolerance
        # is_height_sim = abs(box['h'] - std_h) <= tolerance
        is_width_sim =  box['w'] > 15
        is_height_sim = box['h'] > 15

        pt1_a = (box['x'] - margin, box['y'] + margin)
        pt2_a = (box['x'] - margin + box['w'], box['y'] + box['h'] + margin)

        if is_width_sim and is_height_sim:
            standard_checkboxes.append(box)
            cv2.rectangle(original, pt1_a, pt2_a, (0, 255, 0), thickness=1)
        else:
            unusual_checkboxes.append(box)
            # cv2.rectangle(original, pt1_a, pt2_a, (0, 0, 255), thickness=1)

    debug_file = f"debug_boxes.jpg"
    cv2.imwrite(debug_file, original)
    print(f" Debug image saved as {debug_file} ---", file=sys.stderr)

    print(f"Checked boxes - Standard: {len(standard_checkboxes)} | Unusual: {len(unusual_checkboxes)}")
    return standard_checkboxes, unusual_checkboxes


def crop_at_first_large_gap(roi, gap_threshold=15):
    """
    Crops the ROI at the first vertical gap wider than gap_threshold.
    Assumes ink is white (255) and background is black (0).
    """
    # 1. Project the ink onto the X-axis (any column with a white pixel is > 0)
    projection = np.sum(roi, axis=0)

    # 2. Find indices where the column is empty (black)
    empty_cols = np.where(projection == 0)[0]

    if len(empty_cols) == 0:
        return roi  # No gaps found

    # 3. Find groups of consecutive empty columns
    # We look for a jump in indices to find the start of the first big gap
    gap_count = 0
    for i in range(1, len(empty_cols)):
        if empty_cols[i] == empty_cols[i - 1] + 1:
            gap_count += 1
            if gap_count >= gap_threshold:
                # We found a gap wide enough to be a "break"
                # Return the image up until the start of this gap
                break_point = empty_cols[i - gap_count]
                return roi[:, 0:break_point]
        else:
            gap_count = 0

    return roi


def crop_at_first_large_gap_inverse2(roi, gap_threshold=15, noise_tolerance=2):
    """
    Crops ROI at the first vertical gap for Black Ink on White Background.
    noise_tolerance: Max number of 'ink' (dark) pixels allowed in a gap column.
    """
    # 1. Convert to a binary "ink" map where ink=1, paper=0
    # We treat any pixel darker than 127 as ink
    _, ink_map = cv2.threshold(roi, 127, 1, cv2.THRESH_BINARY_INV)

    # 2. Project the ink onto the X-axis
    # Each value in 'projection' is the count of dark pixels in that column
    projection = np.sum(ink_map, axis=0)

    # 3. Find indices where the column is 'mostly' white paper
    # (i.e., contains very few dark pixels)
    empty_cols = np.where(projection <= noise_tolerance)[0]

    if len(empty_cols) == 0:
        return roi

    # 4. Find consecutive groups of empty columns
    gap_count = 0
    for i in range(1, len(empty_cols)):
        if empty_cols[i] == empty_cols[i - 1] + 1:
            gap_count += 1
            if gap_count >= gap_threshold:
                # Calculate the start of the gap
                break_point = empty_cols[i - gap_count]

                # Safety: If the gap is at the very beginning, keep looking
                if break_point < 5:
                    continue

                return roi[:, 0:break_point]
        else:
            gap_count = 0

    return roi


def crop_at_first_large_gap_inverse(roi, gap_threshold=15, noise_tolerance=2):
    """
    Crops ROI at the first vertical gap AFTER finding ink.
    Designed for Black Ink on White Background.
    """
    # 1. Convert to a binary "ink" map (ink=1, paper=0)
    _, ink_map = cv2.threshold(roi, 127, 1, cv2.THRESH_BINARY_INV)

    # 2. Project ink onto the X-axis
    projection = np.sum(ink_map, axis=0)

    # 3. Iterate through columns to find the first text and then the first gap
    found_ink = False
    gap_count = 0

    for x, ink_count in enumerate(projection):
        # State A: Looking for the start of the text
        if not found_ink:
            if ink_count > noise_tolerance:
                found_ink = True
            continue  # Keep skipping until text is found

        # State B: Text has been found, now looking for a significant gap
        if ink_count <= noise_tolerance:
            gap_count += 1
            if gap_count >= gap_threshold:
                # We found a real gap after text.
                # Crop at the point where the gap started.
                break_point = x - gap_count + 1
                return roi[:, 0:break_point]
        else:
            # We hit more ink, so reset the gap counter
            gap_count = 0

    # Return original if no large gap was found after the text
    return roi

def extract_checkbox_labels(original_image, preprocessed_image, checkboxes, zone, debug_draw):
    """Extract text labels associated with checkboxes"""
    results = []

    if debug_draw:
        original_image = original_image.copy()  # make in-memory copy

    for cb in checkboxes:
        # Define region to the right of checkbox for label
        if zone == 0:   # label on right
            label_x = cb['x'] + cb['w'] + 5
            label_y = cb['y'] - 5
            label_w = 300  # Adjust based on your form
            label_h = cb['h'] + 10
        elif zone == 1:  # short near label on left
            width = 310
            label_x = cb['x'] + cb['w'] - width
            label_y = cb['y'] - 5
            label_w = width  # Adjust based on your form
            label_h = cb['h'] + 10
        else:  # short far label on left
            width = 500
            label_x = cb['x'] + cb['w'] - width
            label_y = cb['y'] - 5
            label_w = width  # Adjust based on your form
            label_h = cb['h'] + 10

        # Ensure within bounds
        h, w = preprocessed_image.shape[:2]
        label_x = max(0, label_x)
        label_y = max(0, label_y)
        label_w = min(label_w, w - label_x)
        label_h = min(label_h, h - label_y)

        # Extract label region
        label_roi = preprocessed_image[label_y:label_y + label_h, label_x:label_x + label_w]

        # crop if large gap between words to eliminate bleeding into neighboring label.
        label_roi_single_word = crop_at_first_large_gap_inverse(label_roi, gap_threshold=10)

        # OCR on label
        custom_config = r'--oem 3 --psm 7'  # Single line mode
        label_text = pytesseract.image_to_string(label_roi_single_word, config=custom_config).strip()

        # Remove non-alphanumeric characters
        # [^a-zA-Z0-9] means "anything that is NOT a letter or a number"
        clean_text = re.sub(r'[^a-zA-Z0-9]', '', label_text)

        if len(clean_text) == 0:
            label_text = pytesseract.image_to_string(label_roi, config=custom_config).strip()
            clean_text = re.sub(r'[^a-zA-Z0-9]', '', label_text)
            clean_text = re.sub(r'(.+?)[A-Z].*', r'\1', clean_text) # Remove trailing appended Capital word,  FaceCc -> Face

        results.append({
            'checkbox_position': (cb['x'], cb['y']),
            'checked': cb['checked'],
            'label': clean_text,
            'confidence': cb['fill_percentage']
        })

        if debug_draw:
            # 1. First Rectangle (Red) round checkbox
            margin = 1
            pt1_a = (cb['x']-margin, cb['y']+margin)
            pt2_a = (cb['x']-margin + cb['w'], cb['y'] + cb['h']+margin)
            ## cv2.rectangle(original_image, pt1_a, pt2_a, (0, 0, 255), thickness=1)

            # 2. Second Rectangle (Green) Label
            pt1_b = (label_x, label_y)
            pt2_b = (label_x + label_w, label_y + label_h)
            ## cv2.rectangle(original_image, pt1_b, pt2_b, (0, 255, 0), thickness=1)

            # --- 3rd Rectangle Logic ---
            # Find the extreme corners that span both
            min_x = min(pt1_a[0], pt1_b[0])
            min_y = min(pt1_a[1], pt1_b[1])
            max_x = max(pt2_a[0], pt2_b[0])
            max_y = max(pt2_a[1], pt2_b[1])

            # Apply 2-pixel margin (expanding outward)
            margin = 2
            final_pt1 = (min_x - margin, min_y - margin)
            final_pt2 = (max_x + margin, max_y + margin)

            # Draw the spanning rectangle (Yellow)
            cv2.rectangle(original_image, final_pt1, final_pt2, (0, 255, 0), thickness=2)

    if debug_draw:
        if zone == 0:
            debug_file = f"debug_label_right.jpg"
        elif zone == 1:
            debug_file = f"debug_label_left1.jpg"
        else:
            debug_file = f"debug_label_left2.jpg"

        cv2.imwrite(debug_file, original_image)
        print(f" Debug image saved as {debug_file} ---", file=sys.stderr)

    print(f" Extracted {len(results)} labels")
    return results

def extract_from_xfa(pdf_path):
    """Process entire form"""
    # Preprocess
    preprocessed, original = preprocess_for_checkbox_detection(pdf_path)

    # Remove lines (optional - test if it helps)
    # preprocessed is already a grayscale numpy array, convert to PIL for remove_lines
    no_lines = remove_lines(Image.fromarray(preprocessed))
    print(f" removed {no_lines.size} lines")

    # no_lines is already grayscale, no need to convert
    # If it's a PIL Image, convert it:
    if isinstance(no_lines, Image.Image):
        no_lines_cv = np.array(no_lines)
    else:
        # It's already a numpy array
        no_lines_cv = no_lines

    debug_draw = True

    idx = 0
    results = []
    # Detect checkboxes
    # all_checkboxes = detect_all_checkboxes(no_lines_cv)
    # ischecked, unchecked =  detect_checkboxes_adaptive(original, no_lines_cv)

    ischecked  = detect_checkboxes_by_region(original, no_lines_cv, 0)
    results = extract_checkbox_labels(original, preprocessed, ischecked, 0, debug_draw)
    print("Labels on right")
    idx = display_results(idx, results)

    ischecked = detect_checkboxes_by_region(original, no_lines_cv, 1)
    results = extract_checkbox_labels(original, preprocessed, ischecked, 1, debug_draw)
    print("Labels on near left")
    idx = display_results(idx, results)

    ischecked = detect_checkboxes_by_region(original, no_lines_cv, 2)
    results = extract_checkbox_labels(original, preprocessed, ischecked, 2, debug_draw)
    print("Labels on far left")
    idx = display_results(idx, results)

    return results


def display_results(idx, results):
    for item in results:
        idx += 1
        print(f"{idx:3d}, {item['checkbox_position']}, {item['label']}")

    return idx

def template_matching_approach(pdf_path, template_path):
    """If forms are identical, use template matching"""
    # Load template (crop a checkbox from a clean form)
    template = cv2.imread(template_path, 0)

    # Convert PDF
    images = convert_from_path(pdf_path, dpi=300)
    img = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2GRAY)

    # Match
    result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    locations = np.where(result >= threshold)

    return list(zip(*locations[::-1]))

def diagnose_pdf_conversion(pdf_path):
    """Diagnose PDF conversion issues"""
    print(f"Checking PDF path: {pdf_path}")
    # print(f"   File exists: {os.path.exists(pdf_path)}")
    print(f"   File size: {os.path.getsize(pdf_path) if os.path.exists(pdf_path) else 'N/A'} bytes")



def check_dependencies():
    print("Checking dependencies...\n")

    # Check Python packages
    packages = ['cv2', 'numpy', 'PIL', 'pdf2image', 'pytesseract']
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"✓ {pkg} installed")
        except ImportError:
            print(f"✗ {pkg} NOT installed - run: pip install {pkg if pkg != 'PIL' else 'Pillow'}")

    # Check system tools
    tools = ['tesseract', 'pdftoppm']
    for tool in tools:
        path = shutil.which(tool)
        if path:
            print(f"✓ {tool} found at {path}")
        else:
            print(f"✗ {tool} NOT found - run: brew install {tool if tool == 'tesseract' else 'poppler'}")


# Use it
check_dependencies()
input_pdf = '/Users/dennislang/opt/projects/projects-python/image.pdf'
images = diagnose_pdf_conversion(input_pdf)

results = extract_from_xfa(input_pdf)


print("---done---", file=sys.stderr)