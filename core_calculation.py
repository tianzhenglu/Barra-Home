"""this file is used for calculating descriptors' weights, factor return and etc"""
from typing import Union
from matplotlib import pyplot as plt
from scipy.stats import norm
import seaborn as sns
import numpy as np
import pandas as pd
import math


def calc_weights(descriptors: pd.DataFrame):
    pass


def calc_factor_returns():
    pass


def get_ESTU_industry_universe(ESTU: pd.DataFrame, path_industry: str):
    # Reduce size of industry info matrix to number of stocks in ESTU
    # Code in File should in following format: ######.AA
    industry_portfolio_df = pd.read_csv(path_industry, encoding='gbk',index_col = 1)
    estu_array = ESTU.melt().iloc[:,1].drop_duplicates().dropna()
    industry = industry_portfolio_df['industry']
    industry_estu = industry[~industry.index.duplicated()].reindex(estu_array)
    return industry_estu


def calc_industry_matrix(weekly_ESTU: pd.Series,industry_estu: pd.Series):
    # Calculate weekly industry matrix from weekly ESTU and industry ESTU
    weekly_ESTU = weekly_ESTU.dropna()
    industry_estu = industry_estu.reindex(weekly_ESTU).to_frame()
    industry_estu['identity_col'] = 1
    industry_matrix = industry_estu.pivot(columns = 'industry',values = 'identity_col')
    return industry_matrix


def calc_industry_weights(industry_distribution: pd.DataFrame, weekly_value: pd.Series):
    '''
    Input: 
        industry_distribution: stock's belongings of industry. 1  is in, Nan is out
        weekly_value: each stock's total value at certain time
        
    Output:
        weights_df: each industry's value percentage of total market
    '''
    industry_distribution = industry_distribution.fillna(0)
    industry_distribution = industry_distribution.astype(int)
    weekly_value = weekly_value.astype(float)
    industry_weekly_total = np.dot(industry_distribution,weekly_value)
    industry_sum = industry_weekly_total.sum(axis = 0)
    weights_df = np.divide(industry_weekly_total,industry_sum)
    weights_df = pd.Series(data=weights_df, index=industry_distribution.index, name=weekly_value.name)
    return weights_df
    

def build_restricted_matrix(initial_df: pd.DataFrame, weights_df: pd.Series):
    '''
    Input:
        initial_df: identity matrix, size = 1+p+q
        weights_df: weightings of each industry, size p
        
    Output:
        Modified restricted matrix
    
    '''
    initial_df_np = initial_df.to_numpy()
    weights_df_np = weights_df.to_numpy()
    total_industry = weights_df.size
    weights_mod = -weights_df_np[0:total_industry-1]/weights_df_np[total_industry-1]
    initial_df_np[total_industry,1:total_industry] = weights_mod
    initial_df_np[total_industry,total_industry] = 0
    return initial_df_np


def winsorize(exposure_per_period: Union[np.ndarray, pd.DataFrame]):
    exposure_per_period[exposure_per_period < -3] = 3
    exposure_per_period[exposure_per_period > 3] = 3
    return exposure_per_period


def plot_histogram(exposure_per_period: Union[np.ndarray, pd.Series]):
    # plt.hist(exposure_per_period, bins=20, normed=True, alpha=0.5, histtype='stepfilled',
    #          color='steelblue', edgecolor='none')
    # plt.title("histogram")
    sns.set()
    sns.distplot(exposure_per_period, hist=True, fit=norm, kde=True)
    plt.show()


def standardize(raw_dtors: pd.DataFrame, cir_weights: pd.DataFrame):
    """z_scores"""
    total_mv_per_period = np.nansum(cir_weights.values, axis=1)
    mv_weights_mat = (cir_weights.values.T / total_mv_per_period).T
    mv_weighted_mat = np.multiply(raw_dtors.values, mv_weights_mat)
    mu_per_period = np.nansum(mv_weighted_mat, axis=1)
    std = np.nanstd(raw_dtors.values, axis=1)
    std_dtors = ((raw_dtors.values.T - mu_per_period)/std).T
    return std_dtors


def validate_factors():
    pass


def build_weights_matrix_per_week(mv_df: pd.Series):
    scale_data = []
    for symbol in mv_df.index:
        scale_data.append(mv_df.at[symbol]/mv_df.sum())
    adj_weights = np.sqrt(scale_data)/np.sqrt(scale_data).sum()
    v_mat = pd.DataFrame(np.diag(adj_weights), index=mv_df.index, columns=mv_df.index)
    return v_mat


def sum_with_weight_srs(data, weight):
    data_df = pd.concat(data, axis=1)
    values=np.ma.average(np.ma.masked_array(data_df, np.isnan(data_df)), axis=1, weights=weight).filled(np.nan)
    return pd.Series(values, index=data_df.index)


def orthogonalize(y, x, **kwargs):
    x_tmp = x[~(np.isnan(x))|np.isnan(y)]
    y_tmp = y[~(np.isnan(x))|np.isnan(y)]
    sumx2 = sum(x_tmp**2)
    sumx = sum(x_tmp)
    sumy = sum(y_tmp)
    sumxy = sum(y_tmp*x_tmp)
    n = len(x_tmp)

    frac = n*sumx2 - sumx**2
    if frac == 0:
        return np.array([np.nan for i in range(len(y))])
    else:
        a = (sumx2 * sumy - sumx * sumxy) / frac
        b = (n * sumxy - sumx * sumy) / frac
        e = y - a - b * x
    return e


if __name__ == '__main__':
    industry_distribution = pd.read_csv('./test_set/test_industry_2.csv', index_col=0)
    weekly_total = pd.read_csv('./test_set/total_mv_test_set.csv', index_col=0)
    total1 = weekly_total.iloc[1]
    weights_df = calc_industry_weights(industry_distribution, total1)
    initial_df = pd.DataFrame(data=np.identity(10))
    a = build_restricted_matrix(initial_df, weights_df)