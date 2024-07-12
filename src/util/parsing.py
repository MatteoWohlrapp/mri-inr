"""
Utils function for parsing filenames.
"""

import re


def parse_filename(filename: str) -> str:
    """
    Parse the filename to extract the MRI type and the hex values.

    Args:
        filename (str): The filename to parse.

    Returns:
        str: The parsed filename.
    """
    match = re.match(r"file_(\w+)_AX(\w+)_([0-9]+)_([0-9]+)", filename)
    if match:
        mri_type = match.group(2).lower()
        hex1 = hex(int(match.group(3)))[2:]
        hex2 = hex(int(match.group(4)))[2:]
        return f"{match.group(1)}_{mri_type}_{hex1}_{hex2}"
    else:
        return None
