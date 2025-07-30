import os
import random

from Finance.Thesis.functions import make_json_from_summary, make_summary

# Paths
source_dir = r"D:\Files\OneDrive - University of Luxembourg\Thesis\PreqinDownloads\Investement Reports"
summary_dir = r"C:\Users\leocr\Projects\Economics\Finance\Thesis\files\responses\summaries"

# Ensure output directory exists
os.makedirs(summary_dir, exist_ok=True)

# Get all PDF files from the source directory
all_pdfs = [f for f in os.listdir(source_dir) if f.lower().endswith('.pdf')]

# Randomly select 50 PDFs
n=50
#n=len(all_pdfs)

selected_pdfs = random.sample(all_pdfs, min(n, len(all_pdfs)))

make_summary(source_dir,summary_dir,selected_pdfs)