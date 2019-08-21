import pandas as pd
import xlrd as xl

britons = pd.ExcelFile('britons.xlsx').parse()
for column in britons:
    print(column)
