import os, sys; sys.path.append(os.getcwd())

from common.dataset.journalist_dataset import create_journalist_dataset

print(create_journalist_dataset())