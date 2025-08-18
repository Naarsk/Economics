from Finance.Thesis.functions import make_excel_from_json

#Paths
json_dir = r"C:\Users\leocr\Projects\Economics\Finance/Thesis/files/responses\parsed_json_21_TPG"
excel_dir = r"C:\Users\leocr\Projects\Economics\Finance/Thesis/files/responses\excel"
excel_name = "21_TPG.xlsx"

#Process all jsons
make_excel_from_json(json_dir, excel_dir, excel_name)
