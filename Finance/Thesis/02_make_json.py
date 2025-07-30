import os
from Finance.Thesis.functions import make_json_from_summary

# Paths
summary_dir = r"C:\Users\leocr\Projects\Economics\Finance\Thesis\files\responses\summaries"
output_dir = r"C:\Users\leocr\Projects\Economics\Finance\Thesis\files\responses\parsed_json"

# Ensure output directory exists
os.makedirs(summary_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Process all summaries
make_json_from_summary(summary_dir,output_dir)
