from Finance.Thesis.B_analysis_functions import make_excel_from_json

fund_number_name = "22_TCG"
#Paths
json_dir = fr"C:\Users\leocr\Projects\Economics\Finance/Thesis/files/responses\parsed_json_{fund_number_name}"
excel_dir = r"C:\Users\leocr\Projects\Economics\Finance/Thesis/files/responses\excel"
excel_name = f"{fund_number_name}.xlsx"

#Process all jsons
make_excel_from_json(json_dir, excel_dir, excel_name)
