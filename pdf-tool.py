#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
import shutil

def find_chrome():
    """
    Attempts to locate the Google Chrome executable.
    """
    # Common macOS paths
    macos_paths = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        os.path.expanduser("~/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    ]
    for path in macos_paths:
        if os.path.exists(path):
            return path

    # Check PATH for linux/windows
    for cmd in ["google-chrome", "chrome", "chromium", "chromium-browser"]:
        path = shutil.which(cmd)
        if path:
            return path
            
    # Common Windows paths
    windows_paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
    ]
    for path in windows_paths:
        if os.path.exists(path):
            return path
            
    return None

def chrome_convert_xfa(input_path, output_path):
    chrome_exe = find_chrome()
    if not chrome_exe:
        print("Error: Google Chrome executable not found. Chrome is required for XFA rendering.")
        sys.exit(1)

    abs_input = os.path.abspath(input_path)
    abs_output = os.path.abspath(output_path)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(abs_output), exist_ok=True)

    # Command to print to PDF
    # --no-pdf-header-footer removes the timestamp and URL from the page
    cmd = [
        chrome_exe,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer", 
        f"--print-to-pdf={abs_output}",
        f"file://{abs_input}"
    ]

    print(f"Converting '{input_path}' to '{output_path}' using Chrome...")
    try:
        # capture_output=True requires Python 3.7+
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if os.path.exists(abs_output):
            print(f"Success! Output saved to: {output_path}")
        else:
            print("Error: Chrome command finished but output file was not created.")
            
    except subprocess.CalledProcessError as e:
        print(f"Error running Chrome: {e}")
        if e.stderr:
            print(f"Stderr: {e.stderr.decode()}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main():
    """
    Parses command-line arguments and initiates the PDF flattening process using Headless Chrome.
    """
    parser = argparse.ArgumentParser(description="Flatten a dynamic XFA PDF to a static PDF using Headless Chrome.")
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

    chrome_convert_xfa(input_file, output_file)

if __name__ == "__main__":
    main()
