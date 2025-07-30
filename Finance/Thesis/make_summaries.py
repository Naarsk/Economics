import os

from Finance.Thesis.query_ollama import query_ollama, extract_pdf_text, build_prompt


def make_summary(source_dir,output_dir,selected_pdfs):


    # Process each PDF
    for pdf_file in selected_pdfs:
        pdf_path = os.path.join(source_dir, pdf_file)
        text = extract_pdf_text(pdf_path)
        final_prompt = build_prompt(text)

        # 3. Query DeepSeek for strict JSON
        response = query_ollama(prompt=final_prompt, model="deepseek-r1")

        # Save response to a .txt file
        output_filename = os.path.splitext(pdf_file)[0] + "_summary.txt"
        output_path = os.path.join(output_dir, output_filename)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(response)

        print(f"Processed: {pdf_file} → {output_filename}")
    print("All selected documents have been processed.")



