import os
import shutil
import re

# Specify the folder containing the PDF files
source_folder = 'E:\\derivingStructure\\ori_data\\18-23_CSR_REPORT'

# Get all files in the folder
files = os.listdir(source_folder)

# Regular expression to match the year
year_pattern = re.compile(r'_(\d{4})年度')

for file in files:
    if file.endswith('.pdf'):
        # Search for the year in the filename
        match = year_pattern.search(file)
        if match:
            year = match.group(1)
            # Create the year folder if it doesn't exist
            year_folder = os.path.join(source_folder, year)
            if not os.path.exists(year_folder):
                os.makedirs(year_folder)
            # Move the PDF file to the corresponding year folder
            shutil.move(os.path.join(source_folder, file), os.path.join(year_folder, file))