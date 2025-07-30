from Finance.Thesis.functions import make_excel

#Paths
json_dir = r"C:\Users\leocr\Projects\Economics\Finance\Thesis\files\responses\parsed_json"
excel_dir = r"C:\Users\leocr\Projects\Economics\Finance\Thesis\files\responses\excel"
excel_name = "clean_outlooks.xlsx"

#Process all jsons
make_excel(json_dir, excel_dir, excel_name)
