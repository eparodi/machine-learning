import pandas as pd

britons = pd.ExcelFile('britons.xlsx').parse()
probs = britons.groupby('Nacionalidad').mean()
probs_britons = britons.groupby('Nacionalidad').agg({'Nacionalidad': ['count']})
probs_britons = probs_britons.div(probs_britons.sum())

probs = probs.to_dict()
probs_britons = probs_britons.to_dict()
probs_britons = probs_britons[list(probs_britons.keys())[0]]

inp = (1, 0, 1, 1, 0)
keys = list(probs.keys())

for _cls in probs_britons.keys():
    prob = 1
    for x in range(len(keys)):
        i_value = inp[x]
        key = keys[x]
        if i_value:
            prob *= probs[key][_cls]
        else:
            prob *= (1 - probs[key][_cls])
    prob *= probs_britons[_cls]
    print("{_cls}: {prob}".format(_cls=_cls, prob=prob))

