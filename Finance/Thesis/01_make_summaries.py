import os
import random

from Finance.Thesis.A_processing_functions import make_summary

# Paths
source_dir = r"D:\Files\OneDrive - University of Luxembourg\Thesis\PreqinDownloads\GPs\24_BS"
summary_dir = r"C:\Users\leocr\Projects\Economics\Finance\Thesis\files/responses/summaries_24_BS"
url = "http://localhost:11434/api/generate"
model="deepseek-r1"
variable_of_interest = "the outlook on capital distributions"
period = "year following the report date"
fund_manager = "Apollo Global Management"


# Ensure output directory exists
os.makedirs(summary_dir, exist_ok=True)

# Get all PDF files from the source directory
all_pdfs = [f for f in os.listdir(source_dir) if f.lower().endswith('.pdf')]

# Randomly select n PDFs

n=len(all_pdfs)
selected_pdfs = random.sample(all_pdfs, min(n, len(all_pdfs)))

make_summary(source_dir, summary_dir, selected_pdfs, url=url, model=model)