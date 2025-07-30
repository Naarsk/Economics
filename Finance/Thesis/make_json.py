# Process all summaries
from Finance.Thesis.from_summaries import make_json_from_summary
from Finance.Thesis.make_summaries import make_summary
import os
import random

# Paths
source_dir = r"D:\Files\OneDrive - University of Luxembourg\Thesis\PreqinDownloads\Investement Reports"
summary_dir = r"C:\Users\leocr\Projects\Economics\Finance\Thesis\files\responses\summaries"
output_dir = r"C:\Users\leocr\Projects\Economics\Finance\Thesis\files\responses\parsed_json"

# Ensure output directory exists
os.makedirs(summary_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Get all PDF files from the source directory
all_pdfs = [f for f in os.listdir(source_dir) if f.lower().endswith('.pdf')]

# Randomly select 50 PDFs
n=len(all_pdfs)
#selected_pdfs = random.sample(all_pdfs, min(n, len(all_pdfs)))
selected_pdfs = all_pdfs

#make_summary(source_dir,summary_dir,selected_pdfs)
make_json_from_summary(summary_dir,output_dir)
