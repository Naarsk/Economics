json_schema = """{
  "results": {
    "fund_name": "<fund name>",
    "report_date": "<yyyymmdd>",
    "outlook": "<increase|stable|decrease>",
    "magnitude": <percentage>,
    "confidence": <number between 0 and 1>,
    "explanation": "<short one-line reasoning>"
  }
}"""

variable_of_interest = "the outlook on capital distributions"

period = "year following the report date"

fund_manager = "The Carlyle Group"

def build_summary_prompt(text):
    """
    Builds the final prompt for DeepSeek to extract the outlook
    strictly in the required JSON format.
    """
    return f"""
            You are a financial analyst.  
            Your task: summarize the following investment report, 
            keep track of the name of the fund and the date of the report, 
            focus on {variable_of_interest} for the {period} in the perspective of fund manager {fund_manager}, its magnitude in percentage terms, and your confidence in the prediction.
            
            Context from the document:
            \"\"\"{text}\"\"\"
            
            remember your task: summarize the following investment report, 
            keep track of the name of the fund and the date of the report, 
            focus on {variable_of_interest} for the {period} in the perspective of fund manager {fund_manager}, its magnitude in percentage terms, and your confidence in the prediction.
            """


def build_json_prompt(summary_text):
    return f"""
        You are a financial analyst that must output only a valid JSON object, forecasting {variable_of_interest} for the {period}.
        
        Context from the document:
        \"\"\"{summary_text}\"\"\"
        
        Based on the above context, produce strictly one JSON object following this schema:
        {json_schema}
        
        Respond with ONLY the JSON object.
        """
