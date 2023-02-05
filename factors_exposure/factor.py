import pandas as pd
from typing import Dict
import numpy as np
#import core_calculation as calc

#from descriptor import Descriptor


class Factor(object):
    weights = None

    def __init__(self, name: str, descriptors_dict: Dict):
        self.name = name
        self.descriptors_dict = descriptors_dict
        self.factor_exposure = None


if __name__ == '__main__':
    lncap = Descriptor('LNCAP', None)

    def init_mv_estu_df():
        df = pd.read_csv('../../../data_set/factors_data/total_mv_estu.csv', index_col=0)
        return df

    mv_estu_df = init_mv_estu_df()
    lncap_raw_dtors_df = mv_estu_df.apply(lambda val: np.log(val))
    lncap.load_data(lncap_raw_dtors_df)

    lncap_expo_standardized = calc.standardize(lncap.raw_exposure, mv_estu_df)
    test_array = lncap_expo_standardized[3]
    calc.plot_histogram(test_array)
    lncap_expo_standardized = calc.winsorize(lncap_expo_standardized)
    test_array_winsorized = lncap_expo_standardized[3]
    calc.plot_histogram(test_array_winsorized)

    def init_mv_total_market_df():
        df = pd.read_csv('../../../data_set/factors_data/total_mv.csv', index_col=0)
        return df

    mv_market_df = init_mv_total_market_df()
    lncap_raw_dtors_total_market_df = mv_market_df.apply(lambda val: np.log(val))

    lncap_raw_dtors_total_market_df = calc.standardize(lncap_raw_dtors_total_market_df, mv_market_df)
    test_total_market_array = lncap_raw_dtors_total_market_df[4]
    median = np.nanmedian(test_total_market_array)
    test_total_market_array = test_total_market_array[~np.isnan(test_total_market_array)]
    calc.plot_histogram(test_total_market_array)

    beta = Factor('Beta', None)


