from typing import List
import pandas as pd


class EstimationUniverse(object):
    def __init__(self, universe_matrix: pd.DataFrame):
        self.universe_matrix = universe_matrix

    def prune(self):
        """
        kick out all ST ST** stocks
        """
        pass


class CoverageUniverse(object):
    def __init__(self, stock_list: List):
        self.stocks = stock_list


if __name__ == '__main__':
    universe_df = pd.read_csv('./test_set/test_estimation_universe.csv', index_col=0)
    estimation_universe = EstimationUniverse(None)
    estimation_universe.load_estimation_universe(universe_df)