json_schema = """{
  "results": {
    "name": "<fund name>",
    "date": "<yyyymmdd>",
    "outlook": "<increase|stable|decrease>",
    "magnitude": <percentage>,
    "confidence": <number between 0 and 1>,
    "explanation": "<short one-line reasoning>"
  }
}"""

variable_of_interest = "the outlook on capital distributions"

period = "year following the report date"


def build_summary_prompt(text):
    """
    Builds the final prompt for DeepSeek to extract the outlook
    strictly in the required JSON format.
    """
    return f"""
            You are a financial analyst.  
            Your task: summarize the following investment report, focus on the name of the fund, the date of the report, {variable_of_interest} for the {period}, its magnitude in percentage terms, and your confidence in the prediction.
            
            Context from the document:
            \"\"\"{text}\"\"\"
            
            remember your task: summarize the provided investment report, focus on the name of the fund, the date of the report, {variable_of_interest}for the {period}, its magnitude in percentage terms, and your confidence in the prediction.

            """


def build_json_prompt(summary_text):
    return f"""
        You are a financial analyst that must output only a valid JSON object, forecasting {variable_of_interest} in the next year.
        
        Context from the document:
        \"\"\"{summary_text}\"\"\"
        
        Based on the above context, produce strictly one JSON object following this schema:
        {json_schema}
        
        Respond with ONLY the JSON object.
        """
