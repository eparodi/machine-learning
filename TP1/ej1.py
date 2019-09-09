from common.naive_bayes import NaiveBayes

probs = {
    'P1': {
        'Jóvenes': 0.95,
        'Viejos': 0.03,
    },
    'P2': {
        'Jóvenes': 0.05,
        'Viejos': 0.82,
    },
    'P3': {
        'Jóvenes': 0.02,
        'Viejos': 0.34,
    },
    'P4': {
        'Jóvenes': 0.2,
        'Viejos': 0.92,
    },
}

class_probs = {
    'Jóvenes': 0.1,
    'Viejos': 0.9
}

inp = (1, 0, 1, 0)
bayes = NaiveBayes(probs, class_probs)
prob = bayes.get_probabilities(inp)
print(prob)

probConjunta = prob["Jóvenes"] + prob["Viejos"]
probDatos = prob["Jóvenes"] / probConjunta
print(probDatos)