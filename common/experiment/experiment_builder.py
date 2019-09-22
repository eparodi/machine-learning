
def naive_split(trainingPercentage, dataframe):
    middleIndex = int(len(dataframe) * trainingPercentage)
    return (dataframe[:middleIndex], dataframe[middleIndex:])

def random_split(trainingPercentage, dataframe):
    trainingDataframe = dataframe.sample(frac=trainingPercentage)
    testDataframe = dataframe.sample(frac=(1-trainingPercentage))
    return (trainingDataframe, testDataframe)

def random_with_replacement_split(trainingPercentage, dataframe):
    trainingDataframe = dataframe.sample(frac=trainingPercentage, replace=True)
    testDataframe = dataframe.sample(frac=(1-trainingPercentage), replace=True)
    return (trainingDataframe, testDataframe)

def maxDictItem(map):
    maxCat = ""
    maxCatConfidence = 0
    for category, confidence in map.items():
        if(maxCatConfidence < confidence):
            maxCat = category
            maxCatConfidence = confidence
    return maxCat