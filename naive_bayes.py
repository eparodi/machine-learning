from pandas import DataFrame

class NaiveBayes():
    
    def __init__(self, probs, class_probs):
        self.probs = probs
        self.class_probs = class_probs

    @staticmethod
    def apply_laplace_correction(data_frame: DataFrame, class_col: str):
        data_frame = data_frame.groupby(class_col)
        check_zero = data_frame.sum().eq(0).any().any()
        if check_zero:
            sum_frame = data_frame.sum()
            return (sum_frame + 1) / (data_frame.count() + sum_frame.count())
        else:
            return data_frame.mean()

    @staticmethod
    def from_data_frame(data_frame: DataFrame, class_col: str):
        probs = NaiveBayes.apply_laplace_correction(data_frame, class_col)
        probs = probs.to_dict()

        class_probs = data_frame.groupby(class_col).agg({class_col: ['count']})
        class_probs = class_probs.div(class_probs.sum())
        class_probs = class_probs.to_dict()
        class_probs = class_probs[list(class_probs.keys())[0]]
        return NaiveBayes(probs, class_probs)

    def get_probabilities(self, _input):
        keys = list(self.probs.keys())
        result = dict()
        for _cls in self.class_probs.keys():
            prob = 1
            for x in range(len(keys)):
                i_value = _input[x]
                key = keys[x]
                if i_value:
                    prob *= self.probs[key][_cls]
                else:
                    prob *= (1 - self.probs[key][_cls])
            prob *= self.class_probs[_cls]
            result[_cls] = prob
        return result
