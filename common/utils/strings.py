algorithm = "Algorithm"

accuracy_mean = "Accuracy(μ)"
accuracy_std = "Accuracy(σ)"
accuracy = "Accuracy"
correct = "Correct"
correct_mean = "Correct(μ)"
correct_std = "Correct(σ)"

rounds = "Rounds"
precision = "Precision"
recall = "Recall"

f1_mean = "F1-score(μ)"
f1_std = "F1-score(σ)"
f1 = "F1-score"

weight = "Weight"
trueNegatives = "TrueNegatives"
falsePositives = "FalsePositives"
falseNegatives = "FalseNegatives"
truePositives = "TruePositives"

total = "Total"

def one_line_dict(dict:dict, limit=5):
    return str([str(key) + ":" + str(dict[key])[:limit] for key in list(dict.keys())])