"""z-scores"""
import pandas as pd
import numpy as np


class STD(object):
    calc_type = 'cap_weight'
    weights = None

    def __init__(self, calc_type: str, weights: pd.DataFrame):
        self.calc_type = calc_type
        self.weights = weights

    def calc_weights(self, weights_df: pd.DataFrame, calc_type: str):
        pass

    def standardize(self, raw_dtors: pd.DataFrame, cir_weights: pd.DataFrame):
        mu = np.multiply(raw_dtors, cir_weights)
        std = raw_dtors.std(axis=1)
        stdard_dtors = (raw_dtors-mu) / std
        return stdard_dtors


if __name__ == '__main__':
    df = pd.DataFrame(np.array([[85, 68, 90], [82, 63, 88], [84, 90, 78]]), columns=['统计学', '高数', '英语'],
                      index=['张三', '李四', '王五'])
    df['std'] = df.std(axis=1)
    print(df)
