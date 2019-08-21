import pandas as pd

britons = pd.ExcelFile('britons.xlsx').parse()
for column in britons:
    print(column)
