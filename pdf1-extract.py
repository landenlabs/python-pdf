#!/usr/bin/env python3

import pikepdf
import pdfplumber
from lxml import etree  # You may need to: pip install lxml

import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
# from PIL import Image, ImageFilter, ImageEnhance
from PIL import  ImageEnhance

# import pikepdf
def extract_xfa_checkboxes(pdf_path):
    with pikepdf.open(pdf_path) as pdf:
        # 1. Locate the XFA stream
        try:
            xfa = pdf.Root.AcroForm.XFA
        except (AttributeError, KeyError):
            print("No XFA data found in this file.")
            return

        # 2. Extract the XML parts (XFA is usually a list of keys and streams)
        # We want the 'datasets' part which holds the 'checked' values
        xml_data = b""
        for i in range(0, len(xfa), 2):
            if xfa[i] == "datasets":
                xml_data = xfa[i + 1].read_bytes()
                break

        if not xml_data:
            print("Could not find the dataset stream in the XFA.")
            return

        # 3. Parse the XML
        root = etree.fromstring(xml_data)

        # In XFA, checked boxes usually have a value of '1' or the export value (e.g., 'Yes')
        # We search for all nodes that have text content
        print(f"{'Field Name':<40} | {'Value'}")
        print("-" * 55)
        for element in root.iter():
            if element.text and element.text.strip():
                # Filter out container tags, focus on leaf nodes (the data)
                if len(element) == 0:
                    tag_name = element.tag.split('}')[-1]  # Remove XML namespace
                    print(f"{tag_name:<40} | {element.text}")


def extract_all_xfa_fields(pdf_path):
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
            if value:
                print(f"{name:<45} | {value}")


def dump_all_xfa_parts(pdf_path):
    with pikepdf.open(pdf_path) as pdf:
        try:
            xfa = pdf.Root.AcroForm.XFA
        except:
            return

        for i in range(0, len(xfa), 2):
            part_name = str(xfa[i])
            # Extracting the raw XML for each part
            content = xfa[i + 1].read_bytes()

            filename = f"extracted_{part_name}.xml"
            with open(filename, "wb") as f:
                f.write(content)
            print(f"Exported {part_name} to {filename}")

# import pikepdf
def get_checkboxes(file_path):
    with pikepdf.open(file_path) as pdf:
        if not hasattr(pdf, "Root") or "/AcroForm" not in pdf.Root:
            print("No form fields found.")
            return

        fields = pdf.Root.AcroForm.Fields
        for field in fields:
            # Get the field name
            name = field.get("/T", "Unknown")
            # Get the value (/V is the value key in PDF syntax)
            value = field.get("/V", "/Off")
            
            # Checkboxes are usually /Btn (Buttons)
            if field.get("/FT") == "/Btn":
                status = "Checked" if value != "/Off" else "Unchecked"
                print(f"Field: {name} | Status: {status}")


# import pikepdf
def probe_pdf_fields(file_path):
    with pikepdf.open(file_path) as pdf:
        # 1. Check if AcroForm dictionary even exists
        if not hasattr(pdf.Root, "AcroForm"):
            print("Error: This PDF does not appear to contain any form structure (AcroForm or XFA).")
            return

        acroform = pdf.Root.AcroForm

        # 2. Check for XFA (This is likely why your previous script failed)
        if "/XFA" in acroform:
            print("--- XFA Data Found ---")
            # XFA can be a single stream or an array of streams
            xfa_data = acroform.XFA
            # Often the 'datasets' or 'template' contains the checkbox values
            print("This is an XFA form. Use PDFtk 'dump_data_fields' for the easiest extraction,")
            print("or parse the 'pdf.Root.AcroForm.XFA' stream in Python.")

        # 3. Improved AcroForm Field Extraction (Recursive)
        print("\n--- Searching for Standard Fields ---")
        if "/Fields" in acroform:
            for field in acroform.Fields:
                process_field(field)
        else:
            print("No standard /Fields array found.")


def process_field(field, indent=0):
    # Get Field Name (/T)
    name = field.get("/T", "Unnamed")
    # Get Field Value (/V)
    value = field.get("/V", "Off/Empty")
    # Get Field Type (/FT)
    ftype = field.get("/FT", "Unknown")

    if ftype == "/Btn":  # This is a Button (Checkbox/Radio)
        print(f"{'  ' * indent}Checkbox: {name} | Value: {value}")

    # Check for nested child fields (Kids)
    if "/Kids" in field:
        for kid in field.Kids:
            process_field(kid, indent + 1)



# import pdfplumber
def find_checkbox_states(pdf_path):
    # Common characters used for checked/unchecked boxes in flattened PDFs
    # 111/109 are often used in Windings/Dingbats
    check_chars = ["☑", "☒", "■", "✅", "✓", "✔"]
    uncheck_chars = ["☐", "□", "✧", "○"]

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            print(f"--- Page {i + 1} ---")
            print(page.extract_text())  # If this is blank, the PDF is essentially an image.
            words = page.extract_words(extra_attrs=["fontname", "size"])

            for word in words:
                text = word['text']
                # Check if the text looks like a checkbox symbol
                if any(c in text for c in check_chars + uncheck_chars):
                    state = "CHECKED" if any(c in text for c in check_chars) else "EMPTY"

                    # Since names are lost, we find the text nearest to the box
                    # We'll look for text to the right of the symbol
                    surrounding_text = find_label(page, word)

                    print(f"Found {state} box near text: '{surrounding_text}'")

            # Look for small rectangles (rects) that look like checkboxes
            # Typical checkboxes are between 8x8 and 15x15 units
            rects = page.rects
            for rect in rects:
                if 8 <= rect['width'] <= 15 and 8 <= rect['height'] <= 15:
                    # We found a box! Now we check if there is a 'check' inside it.
                    # We do this by looking for 'curves' or 'lines' inside these coordinates.
                    is_checked = False
                    for path in page.curves + page.lines:
                        # Check if the path (the checkmark) is inside the box
                        if (rect['x0'] < path['x0'] < rect['x1'] and
                                rect['top'] < path['top'] < rect['bottom']):
                            is_checked = True
                            break

                    status = "CHECKED" if is_checked else "EMPTY"

                    # Find the text label next to this geometric box
                    label = find_text_near_coord(page, rect['x1'], rect['top'])
                    print(f"Vector Box at {rect['x0']},{rect['top']} | Status: {status} | Label: {label}")

def find_text_near_coord(page, x, y):
    """Finds text to the right of a specific coordinate."""
    words = page.extract_words()
    # Find words on the same line (y) and to the right (x)
    nearby = [w['text'] for w in words if abs(w['top'] - y) < 5 and w['x0'] > x]
    return " ".join(nearby[:3]) # Return first 3 words


def find_label(page, word_obj):
    """Finds text immediately to the right of the checkbox coordinate."""
    all_text = page.extract_words()
    # Look for words on the same vertical line (y) but further right (x)
    labels = [w['text'] for w in all_text
              if abs(w['top'] - word_obj['top']) < 3
              and w['x0'] > word_obj['x1']]
    return " ".join(labels[:5])  # Return the next 5 words as the 'name'


# import cv2
# import numpy as np
# import pytesseract
# from pdf2image import convert_from_path
# from PIL import Image, ImageFilter, ImageEnhance

# --- CONFIGURATION ---
# Point these to your installed paths
pytesseract.pytesseract.tesseract_cmd = r'c:\opt\pdf\Tesseract-OCR\tesseract.exe'
POPPLER_PATH = r'c:\opt\pdf\Poppler-pdf2img-25.12.0-0\poppler-25.12.0\Library\bin'


def process_pdf_ocr(pdf_path):
    # 1. Convert PDF pages to PIL Images
    pages = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)

    for i, page in enumerate(pages):
        print(f"--- Processing Page {i + 1} ---")

        # Convert PIL image to OpenCV format (BGR)
        img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. Thresholding to find shapes
        # We use adaptive thresholding to handle different lighting/scans
        thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)[1]

        # 3. Detect Rectangles (Checkboxes)
        contours, _ = cv2.find_all_contours if hasattr(cv2, 'find_all_contours') else cv2.findContours(
            thresh,  cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            x, y, w, h = cv2.boundingRect(approx)

            # Check if the shape is "square-ish" and the right size for a checkbox
            aspect_ratio = float(w) / h
            if 0.8 <= aspect_ratio <= 1.2 and 15 < w < 50:

                # 4. Check if "Checked" (Is there ink inside?)
                # We crop the box and count non-white pixels
                roi = thresh[y + 2:y + h - 2, x + 2:x + w - 2]
                height, width = roi.shape
                total_pixels = height * width
                non_zero = cv2.countNonZero(roi)

                # If more than 15% of the box is filled, it's likely checked
                is_checked = "CHECKED" if (non_zero / total_pixels) > 0.15 else "EMPTY"

                # 5. Extract Text Label (Look to the right of the box)
                # We crop a wide area to the right of the checkbox
                label_roi = img[y - 5:y + h + 5, x + w:x + w + 400]
                label_text = pytesseract.image_to_string(label_roi).strip()

                if label_text:  # Only print if we found a label
                    print(f"Box: [{is_checked}] | Label: {label_text}")


def process_pdf_with_debug(pdf_path):
    pages = convert_from_path(pdf_path, dpi=300, poppler_path=POPPLER_PATH, grayscale=True)

    for i, page in enumerate(pages):
        img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Binary threshold: makes the image strictly black and white
        # If  boxes aren't being detected: Your checkboxes might have very thin lines.
        # Try decreasing the first number in cv2.threshold(gray, 210, 255, ...) to 180.
        thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)[1]

        ## Adaptive threshold handles varying ink density across the page
        #thresh = cv2.adaptiveThreshold(
        #    gray, 255,
        #    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #    cv2.THRESH_BINARY_INV,
        #    11, 2  # Block size of 11 pixels, constant subtracted of 2
        #)

        # Create a small kernel to bridge tiny gaps in the checkbox lines
        ## kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        ## thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Find all shapes
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            x, y, w, h = cv2.boundingRect(approx)

            aspect_ratio = float(w) / h
            # Narrowing the size to typical checkbox dimensions (adjust if needed)
            if 0.8 <= aspect_ratio <= 1.2 and 15 < w < 60:

                # Analyze the interior of the box
                roi = thresh[y + 4:y + h - 4, x + 4:x + w - 4]
                mask_area = roi.shape[0] * roi.shape[1]
                if mask_area == 0: continue

                fill_level = cv2.countNonZero(roi) / mask_area
                is_checked = fill_level > 0.12  # 12% fill threshold

                # Draw on the image for debugging
                color = (0, 255, 0) if is_checked else (0, 0, 255)  # Green for checked, Red for empty
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

                # OCR the label
                label_roi = gray[y - 5:y + h + 5, x + w:x + w + 500]
                label_text = pytesseract.image_to_string(label_roi, config='--psm 7').strip()

                if label_text:
                    status = "X" if is_checked else " "
                    print(f"[{status}] {label_text}")
                    # Put text on the debug image
                    cv2.putText(img, label_text[:15], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Save the visualization
        debug_file = f"debug_page_{i + 1}.jpg"
        cv2.imwrite(debug_file, img)
        print(f"--- Debug image saved as {debug_file} ---")


def process_chp330_pdf(pdf_path):
    # 1. Higher DPI (300) is critical for small form checkboxes [cite: 15, 32]
    # pages = convert_from_path(pdf_path, dpi=300, poppler_path=POPPLER_PATH, grayscale=True)
    pages = convert_from_path(pdf_path, dpi=300, poppler_path=POPPLER_PATH)

    for i, page in enumerate(pages):
        # Convert PIL to OpenCV format
        img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)[1]

        # # 2. ADAPTIVE THRESHOLDING
        # # Instead of a fixed 20 or 210, this looks at local neighborhoods.
        # # This prevents "losing" columns in the middle if the lighting is uneven.
        # thresh = cv2.adaptiveThreshold(
        #     gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #     cv2.THRESH_BINARY_INV, 11, 2
        # )

        # 3. MORPHOLOGICAL CLOSING
        # Bridges tiny gaps in the checkbox lines often caused by "print-to-image" [cite: 283]
        ## kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        ## thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Find all shapes
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        print(f"--- Processing Page {i + 1} ---")

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h

            # 4. FLEXIBLE BOX DETECTION
            # The CHP 330 has small boxes (Male/Female) and larger ones (Incident Type) [cite: 20, 32]
            # We widen the 'w' range and check for "solidity" (extent)
            if 0.8 <= aspect_ratio <= 1.2 and 20 < w < 80:
                area = cv2.contourArea(cnt)
                extent = float(area) / (w * h)

                # Ensure it's roughly a square and not just a random blob
                if extent > 0.4:
                    # Analyze interior for checkmark (X or check) [cite: 56, 94]
                    roi = thresh[y + 5:y + h - 5, x + 5:x + w - 5]
                    if roi.size == 0: continue

                    fill_level = cv2.countNonZero(roi) / float(roi.shape[0] * roi.shape[1])
                    is_checked = fill_level > 0.10  # Lowered to 10% to catch faint 'X' marks

                    # OCR the label to the right of the checkbox
                    # We grab a wider ROI to capture longer labels like "Unresponsive" [cite: 90]
                    label_roi = gray[y - 5:y + h + 5, x + w:x + w + 450]
                    label_text = pytesseract.image_to_string(label_roi, config='--psm 7').strip()

                    if label_text:
                        status = " [X] " if is_checked else " [ ] "
                        print(f"{status} {label_text}")

                        # Visual Debugging
                        color = (0, 255, 0) if is_checked else (0, 0, 255)
                        cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
                        cv2.putText(img, label_text[:10], (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Save debug image
        cv2.imwrite(f"debug_chp330_page_{i + 1}.jpg", img)


def process_chp330_fast(pdf_path):
    # 1. 200 DPI is a better balance of speed and accuracy for this form
    # pages = convert_from_path(pdf_path, dpi=200)
    pages = convert_from_path(pdf_path, dpi=300, poppler_path=POPPLER_PATH)

    for i, page in enumerate(pages):
        img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. Faster Thresholding (Standard Binary is faster than Adaptive)
        # For a clean PDF/Print, a threshold of 200 works well
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

        # 3. Focus only on the 'Internal' contours to find boxes inside grids
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        if hierarchy is None: continue

        # CHP 330 Specific: The double columns in 'Assessment' and 'Emergency Care'
        # are roughly in the middle-right of the page.
        # We will filter by width/height AND by position to kill false positives.

        width_img = img.shape[1]

        for idx, cnt in enumerate(contours):
            # Only process "child" contours (potential boxes inside a grid)
            if hierarchy[0][idx][3] != -1:
                x, y, w, h = cv2.boundingRect(cnt)

                # Filter 1: Strict Size (at 200 DPI, checkboxes are ~12-25px)
                if 10 < w < 30 and 10 < h < 30:

                    # Filter 2: Position (Ignore the very top and very bottom of the page)
                    if y < 300 or y > 2500: continue

                    # Logic to determine if it's checked
                    roi = thresh[y + 2:y + h - 2, x + 2:x + w - 2]
                    fill_level = cv2.countNonZero(roi) / float(roi.size) if roi.size > 0 else 0
                    is_checked = fill_level > 0.20

                    # Draw only what we find
                    color = (0, 255, 0) if is_checked else (0, 0, 255)
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        cv2.imwrite(f"fast_debug_page_{i + 1}.jpg", img)
        print(f"Page {i + 1} complete.")


# Mapping of Y-coordinate ranges to section types and labels
# These are approximate percentages of page height (0.0 to 1.0)
# to make it scale-independent.
FORM_ZONES = [
    {"name": "Incident", "y_range": (0.10, 0.15), "type": "label_left"},
    {"name": "Sex", "y_range": (0.16, 0.18), "type": "label_left"},
    {"name": "Assessment_Grids", "y_range": (0.23, 0.38), "type": "grid_multi"},
    {"name": "Emergency_Care", "y_range": (0.42, 0.65), "type": "wnl_abn_split"},
]


def process_pdf_fixed_layout(pdf_path):
    # Use 300 DPI for precision
    pages = convert_from_path(pdf_path, dpi=300, grayscale=True, poppler_path=POPPLER_PATH)

    for i, page in enumerate(pages):
        img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        # Consistent Thresholding
        thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)[1]

        # Find boxes
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        extracted_data = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            # Checkbox size filter
            if 0.8 <= (w / h) <= 1.2 and 18 < w < 55:
                # Check state
                roi = thresh[y + 4:y + h - 4, x + 4:x + w - 4]
                is_checked = (cv2.countNonZero(roi) / (roi.size)) > 0.12

                # Identify label based on location
                rel_x = x / width
                rel_y = y / height

                label = "Unknown"

                # LOGIC: Checkbox followed by Label (Most of the form)
                if rel_x < 0.9:  # Exclude margins
                    # Define a small crop to the RIGHT for OCR
                    label_crop = gray[max(0, y - 5):min(height, y + h + 5), x + w:min(width, x + w + 300)]
                    # Use a very restrictive OCR config for single words/short phrases
                    label = pytesseract.image_to_string(label_crop, config='--psm 7').strip()

                # LOGIC: Special handling for the WNL/ABN middle columns
                # These boxes are at specific X-coordinates (approx 0.5 and 0.55)
                if 0.48 < rel_x < 0.58:
                    col_type = "WNL" if rel_x < 0.53 else "ABN"
                    # For these, look to the LEFT for the label
                    label_crop = gray[y - 5:y + h + 5, max(0, x - 250):x]
                    label = f"{pytesseract.image_to_string(label_crop, config='--psm 7').strip()} ({col_type})"

                if label and len(label) > 2:
                    status = "YES" if is_checked else "NO"
                    extracted_data.append({"label": label, "checked": status})

                    # Debug drawing
                    color = (0, 255, 0) if is_checked else (0, 0, 255)
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, label[:10], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        cv2.imwrite(f"final_fix_page_{i + 1}.jpg", img)

        # Print results cleanly
        for item in sorted(extracted_data, key=lambda x: x['label']):
            print(f"{item['label']}: {item['checked']}")


def deskew_image(image):
    """Detects the skew angle of the form and rotates it back to level."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Use Canny to find edges of the form lines
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

    angles = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            # We only care about near-horizontal lines
            if -10 < angle < 10:
                angles.append(angle)

    if not angles: return image

    median_angle = np.median(angles)
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def process_chp330_robust(pdf_path):
    pages = convert_from_path(pdf_path, dpi=300, grayscale=True, poppler_path=POPPLER_PATH)

    for i, page in enumerate(pages):
        # Initial image prep
        img_raw = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)

        # 1. FIX SKEW
        img = deskew_image(img_raw)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)[1]
        height, width = gray.shape

        # Find Boxes
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            rel_x = x / width
            rel_y = y / height

            # Filtering for typical checkbox size on this form
            if 0.8 <= (w / h) <= 1.25 and 18 < w < 60:
                # Determine Check State
                roi = thresh[y + 4:y + h - 4, x + 4:x + w - 4]
                is_checked = (cv2.countNonZero(roi) / float(roi.size)) > 0.12 if roi.size > 0 else False

                # 2. DYNAMIC LABEL DETECTION
                # If box is in the Middle columns (WNL/ABN Assessment Area)
                # Rel_X for WNL is ~0.50, ABN is ~0.55 [cite: 127-141]
                if 0.45 < rel_x < 0.60:
                    column = "WNL" if rel_x < 0.53 else "ABN"
                    # Look LEFT for the text label (e.g., 'Airway', 'Breathing')
                    label_roi = gray[y - 5:y + h + 5, max(0, x - 300):x - 5]
                    label_text = pytesseract.image_to_string(label_roi, config='--psm 7').strip()
                    label_text = f"{label_text} ({column})"

                # If box is in Standard area (Label followed by Box)
                else:
                    # Look RIGHT for label (e.g., 'Off-duty', 'TC')
                    label_roi = gray[y - 5:y + h + 5, x + w + 5:min(width, x + w + 400)]
                    label_text = pytesseract.image_to_string(label_roi, config='--psm 7').strip()

                if len(label_text) > 2:
                    results.append((y, label_text, is_checked))
                    # Debug visuals
                    color = (0, 255, 0) if is_checked else (0, 0, 255)
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        # Sort results by Y coordinate to keep the output in form order
        results.sort(key=lambda x: x[0])
        for _, lab, val in results:
            print(f"[{'X' if val else ' '}] {lab}")

        cv2.imwrite(f"deskewed_debug_page_{i + 1}.jpg", img)


def process_chp330_final_v2(pdf_path):
    # Use 300 DPI for accuracy on small middle boxes
    pages = convert_from_path(pdf_path, dpi=300, grayscale=True, poppler_path=POPPLER_PATH)

    for i, page in enumerate(pages):
        img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        # Binary threshold for box detection
        thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)[1]

        # Use CCOMP hierarchy to find boxes specifically inside grids
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        extracted_data = []

        if hierarchy is not None:
            for idx, cnt in enumerate(contours):
                x, y, w, h = cv2.boundingRect(cnt)
                rel_x, rel_y = x / width, y / height

                # FILTER 1: Strict size for CHP 330 Checkboxes
                if 18 < w < 45 and 18 < h < 45:

                    # FILTER 2: Ignore Headers/Narrative Areas [cite: 1-14, 264-283]
                    if rel_y < 0.14 or rel_y > 0.82:
                        continue

                    # FILTER 3: Refined Check Detection
                    # We crop 25% from each side to look ONLY at the very center
                    # This prevents checkbox borders from triggering a "False Positive"
                    x_pad, y_pad = int(w * 0.25), int(h * 0.25)
                    roi = thresh[y + y_pad: y + h - y_pad, x + x_pad: x + w - x_pad]

                    if roi.size == 0: continue
                    fill_pct = cv2.countNonZero(roi) / float(roi.size)

                    # 15% fill in the center usually indicates a definitive 'X' or mark
                    is_checked = fill_pct > 0.15

                    # FILTER 4: Lane-Based Labeling
                    label_text = ""
                    # Middle columns (WNL / ABN) [cite: 127, 128]
                    if 0.48 < rel_x < 0.58:
                        col = "WNL" if rel_x < 0.53 else "ABN"
                        label_roi = gray[y:y + h, max(0, x - 280):x - 5]
                        label_text = f"{pytesseract.image_to_string(label_roi, config='--psm 7').strip()} ({col})"
                    else:
                        # Standard left-aligned boxes (e.g., Incident, Assessment) [cite: 15-25, 46-104]
                        label_roi = gray[y:y + h, x + w + 5:min(width, x + w + 350)]
                        label_text = pytesseract.image_to_string(label_roi, config='--psm 7').strip()

                    # Filter out short OCR noise
                    clean_label = "".join(e for e in label_text if e.isalnum() or e in " ()").strip()
                    if len(clean_label) > 3:
                        extracted_data.append((y, clean_label, is_checked))
                        # Debugging
                        color = (0, 255, 0) if is_checked else (0, 0, 255)
                        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        # Sort and Print
        extracted_data.sort(key=lambda x: x[0])
        print(f"\n--- PAGE {i + 1} ---")
        for _, lbl, chk in extracted_data:
            print(f"[{'X' if chk else ' '}] {lbl}")

        cv2.imwrite(f"final_debug_p{i + 1}.jpg", img)


# Format: "Label Name": (x, y, width, height)
CHECKBOX_MAP = {
    # Incident Type
    "Off-duty Incident": (2600, 335, 45, 45),
    "TC": (1485, 410, 45, 45),
    "Off-highway accident": (1635, 410, 45, 45),
    "Other Incident": (2080, 410, 45, 45),

    # Patient Info [cite: 32]
    "Male": (1915, 500, 45, 45),
    "Female": (2120, 500, 45, 45),

    # Level of Consciousness [cite: 46, 57, 69, 77, 90]
    "Conscious": (150, 835, 45, 45),
    "Unconscious": (150, 880, 45, 45),
    "Alert": (150, 935, 45, 45),
    "Disoriented": (150, 980, 45, 45),
    "Unresponsive": (150, 1030, 45, 45),

    # Emergency Care - WNL Column [cite: 130-183]
    "Airway (WNL)": (1505, 1275, 40, 40),
    "Breathing (WNL)": (1505, 1345, 40, 40),
    "Circulation (WNL)": (1505, 1415, 40, 40),
    "C-Spine (WNL)": (1505, 1485, 40, 40),
    "Chest (WNL)": (1505, 1555, 40, 40),
    "Abdomen (WNL)": (1505, 1625, 40, 40),
    "Head (WNL)": (1505, 1695, 40, 40),
    "Face (WNL)": (1505, 1765, 40, 40),
    "Back (WNL)": (1505, 1835, 40, 40),
    "Pelvis (WNL)": (1505, 1905, 40, 40),
    "Extremities (WNL)": (1505, 1975, 40, 40),

    # Emergency Care - ABN Column [cite: 130-183]
    "Airway (ABN)": (1675, 1275, 40, 40),
    "Breathing (ABN)": (1675, 1345, 40, 40),
    "Circulation (ABN)": (1675, 1415, 40, 40),
    "C-Spine (ABN)": (1675, 1485, 40, 40),
    "Chest (ABN)": (1675, 1555, 40, 40),
    "Abdomen (ABN)": (1675, 1625, 40, 40),
    "Head (ABN)": (1675, 1695, 40, 40),
    "Face (ABN)": (1675, 1765, 40, 40),
    "Back (ABN)": (1675, 1835, 40, 40),
    "Pelvis (ABN)": (1675, 1905, 40, 40),
    "Extremities (ABN)": (1675, 1975, 40, 40),

    # Medical History [cite: 203, 205, 211, 214, 221, 222, 228, 230]
    "Cardiac": (1170, 2145, 45, 45),
    "Psychiatric": (1450, 2145, 45, 45),
    "Seizure": (1170, 2245, 45, 45),
    "ETOH": (1450, 2245, 45, 45),
    "Diabetes": (1170, 2345, 45, 45),
    "HTN": (1450, 2345, 45, 45),
}


def process_chp330_fixed_map(pdf_path):
    # Ensure DPI is exactly 300 to match the map coordinates
    pages = convert_from_path(pdf_path, dpi=300, grayscale=True, poppler_path=POPPLER_PATH)

    for i, page in enumerate(pages):
        img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # High-sensitivity threshold for ink detection
        thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)[1]

        print(f"\n--- PAGE {i + 1} DATA ---")

        for label, (x, y, w, h) in CHECKBOX_MAP.items():
            # Crop the checkbox area
            # We crop a few pixels INSIDE the border to avoid the frame [cite: 92, 94, 96]
            roi = thresh[y + 8: y + h - 8, x + 8: x + w - 8]

            if roi.size == 0:
                continue

            # Calculate what percentage of the center is 'ink'
            fill_pct = cv2.countNonZero(roi) / float(roi.size)

            # An 'X' or check typically fills > 10% of the internal area
            is_checked = fill_pct > 0.10
            status = "YES" if is_checked else "NO"

            print(f"{label}: {status}")

            # Draw for visual verification
            color = (0, 255, 0) if is_checked else (0, 0, 255)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)

        cv2.imwrite(f"fixed_map_debug_p{i + 1}.jpg", img)


input1_pdf = "c:/opt/projects/projects-python/std.pdf"   # printed (image)
input2_pdf = "c:/opt/projects/projects-python/xfa.pdf"   # xfa
# get_checkboxes(input_pdf)
# probe_pdf_fields(input_pdf)
# extract_xfa_checkboxes(input2_pdf)
# extract_all_xfa_fields(input2_pdf)
# dump_all_xfa_parts(input2_pdf)

# find_checkbox_states(input1_pdf)
# process_pdf_ocr(input1_pdf)
# process_pdf_with_debug(input1_pdf)  # found all boxes, poor text extraction
# process_chp330_pdf(input1_pdf)
# process_chp330_fast(input1_pdf)
# process_pdf_fixed_layout(input1_pdf)
# process_chp330_robust(input1_pdf)
# process_chp330_final_v2(input1_pdf)
process_chp330_fixed_map(input1_pdf)