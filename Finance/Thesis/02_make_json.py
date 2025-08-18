import os
from Finance.Thesis.functions import make_json_from_summary

# arrivato al 72780

# Paths
summary_dir = r"C:\Users\leocr\Projects\Economics/Finance/Thesis/files/responses\summaries_22_TCG"
output_dir = r"C:\Users\leocr\Projects\Economics\Finance/Thesis/files/responses\parsed_json_22_TCG"

# Ensure output directory exists
os.makedirs(summary_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Process all summaries
make_json_from_summary(summary_dir,output_dir)
