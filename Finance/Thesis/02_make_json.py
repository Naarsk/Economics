import os

from Finance.Thesis.A_processing_functions import make_json_from_summary

fund_code = "24_BS"

# Paths
summary_dir = rf"C:\Users\leocr\Projects\Economics/Finance/Thesis/files/responses\summaries_{fund_code}"
output_dir = rf"C:\Users\leocr\Projects\Economics\Finance/Thesis/files/responses\parsed_json_{fund_code}"
url = "http://localhost:11434/api/generate"
model="deepseek-r1"
variable_of_interest = "the outlook on capital distributions"
period = "year following the report date"

# Ensure output directory exists
os.makedirs(summary_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Process all summaries
make_json_from_summary(summary_dir,output_dir, url=url, model=model, variable_of_interest=variable_of_interest,period=period)
