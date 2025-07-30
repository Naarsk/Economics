import os
import re
from Finance.Thesis.query_ollama import query_ollama, build_final_prompt


def remove_think_tags(text):
    """Remove all content enclosed in <think>...</think> tags."""
    text = re.sub(r"```json|```", "", text)
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def make_json_from_summary(summary_dir,output_dir):
    for file in os.listdir(summary_dir):
        if file.lower().endswith(".txt"):
            summary_path = os.path.join(summary_dir, file)

            with open(summary_path, "r", encoding="utf-8") as f:
                summary_text = f.read()

            # ✅ Clean summaries from <think> sections
            cleaned_summary = remove_think_tags(summary_text)

            # Build prompt using the cleaned summary
            prompt = build_final_prompt(cleaned_summary)

            # Query DeepSeek
            response = query_ollama(prompt=prompt, model="deepseek-r1")

            # Save response directly (no JSON cleaning)
            output_file = os.path.splitext(file)[0] + "_parsed.json"
            output_path = os.path.join(output_dir, output_file)

            with open(output_path, "w", encoding="utf-8") as out:
                out.write(remove_think_tags(response))

            print(f"Processed {file} → {output_file}")
    print("✅ All summaries cleaned and processed into JSON.")


