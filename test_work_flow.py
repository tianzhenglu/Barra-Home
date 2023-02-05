import pandas as pd
import numpy as np
import core_calculation as calc
from factors_exposure.beta import Beta
from factors_exposure.country import Country
from factors_exposure.descriptor import Descriptor
from factors_exposure.industry import Industry
from factors_exposure.size import Size
from universe import EstimationUniverse

if __name__ == '__main__':
    'init estimation universe'
    universe_df = pd.read_csv('./test_set/test_estimation_universe.csv', index_col=0)
    estimation_universe = EstimationUniverse(universe_df)
    estimation_universe.universe_matrix = universe_df
    'init country factor exposure'
    country_factor_expo = Country('country', None)
    country_factor_expo.calc_factor_exposure(universe_df.shape[1])

    'init industry factor exposure'
    industry_factor_expo = Industry('industry', None)
    industry_portfolio_df = pd.read_csv('./test_set/test_industry.csv', encoding='gbk', index_col=0)
    industry_portfolio_df.fillna(0, inplace=True)
    industry_factor_expo.calc_factor_expo(industry_portfolio_df.T)

    'init BETA factor exposure'
    beta_factor = Beta('beta', None)

    # index_return matrix
    index_return = pd.read_csv('./test_set/index_daily.csv', index_col=0)
    index_return = index_return.iloc[::-1]
    index_return_rate = (index_return - index_return.shift(1)) / index_return.shift(1)
    # return_rate matrix
    daily_return = pd.read_csv('./test_set/close_price.csv', index_col=0)
    daily_return_rate = (daily_return - daily_return.shift(1)) / daily_return.shift(1)
    columns_list = daily_return.index.tolist()

    # risk_free matrix
    risk_free_mat = pd.read_csv('./test_set/shibor.csv', index_col=0)
    risk_free_mat = risk_free_mat / 100

    w = calc.build_diagonal_mat(10, 4)
    beta_exposure_raw_df = calc.get_beta_matrix(index_return_rate, daily_return_rate, risk_free_mat, w, 10)
    new_df = beta_exposure_raw_df.ewm(axis=1, alpha=0.8, ignore_na=False, adjust=False).mean()
    print(new_df)
    beta_exposure_raw_df.columns = columns_list[10:]

    # standardaize
    mv_df = pd.read_csv('./test_set/total_mv_test_set.csv', index_col=0)


    def cap_weight(rows):
        return rows / rows.sum()


    mv_weighted_df = mv_df.apply(cap_weight, axis=1)
    beta_exposure_raw_weekly_df = beta_exposure_raw_df[mv_df.index.tolist()]
    std_beta_exposure_raw_weekly_mat = calc.standardize(beta_exposure_raw_weekly_df, mv_weighted_df.T)
    std_beta_exposure_raw_weekly_df = pd.DataFrame(std_beta_exposure_raw_weekly_mat,
                                                   columns=beta_exposure_raw_weekly_df.columns,
                                                   index=beta_exposure_raw_weekly_df.index)
    # set factor exposure
    beta_factor.factor_exposure = std_beta_exposure_raw_weekly_df

    'init SIZE factor exposure'
    size_factor = Size('LNCAP', None)

    mv_ln_cap_raw_dtors_df = mv_df.apply(lambda val: np.log(val))
    winsorized_ln_raw_dtors_df = calc.winsorize(mv_ln_cap_raw_dtors_df.T)
    std_size_exposure_raw_weekly_df = calc.standardize(mv_ln_cap_raw_dtors_df.T, mv_weighted_df.T)
    size_factor.factor_exposure = std_size_exposure_raw_weekly_df

    'build v matrix'
    v_mat_dict = {}
    for trade_date in mv_df.index:
        v_mat = calc.build_weights_matrix_per_week(mv_df.loc[trade_date])
        v_mat_dict[trade_date] = v_mat

    print(v_mat_dict)