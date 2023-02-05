#from descriptor import Descriptor
import pandas as pd
import numpy as np
import math
import datetime
from .factor import Factor
from sklearn.linear_model import LinearRegression


class Beta(Factor):
    name = 'BETA'
    path = "G:/torch/prometheus/data/Stock Data/S&P500/"
    ESTU = None
    benchmark = None
    rf = None
    weight = None

    def set_ESTU(self):
        df = pd.read_csv(self.path+"industry.csv")
        self.ESTU = df['Symbol'].to_list()

    def load_daily_price(self,stock:str,start_date:datetime.date,end_date:datetime.date):
        path = self.path+stock+".csv"
        try:
            df = pd.read_csv(path)
        except:
            print("Stock ",stock," not found in database.")
            return None
        df.Date = pd.to_datetime(df['Date']).apply(lambda x:x.date())
        df['Return'] =df.Close.pct_change()
        df = df.loc[(df['Date']>=start_date)&(df['Date']<end_date)]
        return df

    def load_benchmark(self,start_date:datetime.date,end_date:datetime.date):
        self.benchmark = pd.read_csv(self.path+"benchmark.csv")
        self.benchmark.Date = pd.to_datetime(self.benchmark['Date']).apply(lambda x:x.date())
        self.benchmark['Benchmark_Return'] =self.benchmark.Close.pct_change()
        self.benchmark = self.benchmark.loc[(self.benchmark['Date']>=start_date)&(self.benchmark['Date']<end_date)]
        self.benchmark = self.benchmark[['Date','Benchmark_Return']]

    def load_rf(self,start_date:datetime.date,end_date:datetime.date):
        self.rf = pd.read_csv(self.path+"/risk_free.csv")
        self.rf .Date = pd.to_datetime(self.rf['Date']).apply(lambda x:x.date())
        self.rf ['rf'] = (1+self.rf ['Close'])**(1/365)-1 # Daily risk-free rate
        self.rf  = self.rf .loc[(self.rf ['Date']>=start_date)&(self.rf ['Date']<end_date)]
        self.rf  = self.rf [['Date','rf']]
    
    def standardize(self,df,var_name:str):
        std = df[var_name].std()
        mean = df[var_name].mean()
        extreme_outliner = df[(df[var_name]>(mean+5*std))|(df[var_name]<(mean-5*std))].index
        medium_outliner_lower = df[(df[var_name]>(mean-5*std))&(df[var_name]<(mean-3*std))].index
        medium_outliner_upper = df[(df[var_name]<(mean+5*std))&(df[var_name]>(mean+3*std))].index
        df.drop(extreme_outliner,inplace=True)
        for i in medium_outliner_lower.to_numpy():
            df.loc[i,var_name] = mean-std
        for i in medium_outliner_upper.to_numpy():
            df.loc[i,var_name] = mean+std

    def calcBeta_stock(self,stock_name:str,date:datetime.date):
        startdate_m1year = datetime.date(date.year-1,date.month,date.day)
        stock = self.load_daily_price(stock_name,startdate_m1year,date)
        if stock is None:
            return np.nan
        if self.benchmark is None:
            self.load_benchmark(startdate_m1year,date)
        if self.rf is None:
            self.load_rf(startdate_m1year,date)
        merge_data = stock.merge(self.benchmark,on="Date",how='left').merge(self.rf,on='Date',how='left')
        #self.standardize(merge_data,'Return')
        excess_return = (merge_data['Return']-merge_data['rf']).to_numpy().reshape(-1, 1)
        excess_market = (merge_data['Benchmark_Return']-merge_data['rf']).to_numpy().reshape(-1, 1)
        if self.weight is None:
            coeff = np.log(0.5)/63
            self.weight = np.exp(np.arange(len(self.rf),0,-1)*coeff)
        regr = LinearRegression()

        try:
            regr.fit(excess_market,excess_return,self.weight)
        except:
            print("Stock "+stock_name+" is short on data.",len(excess_return),'/',len(self.rf),", Will be eliminated from ESTU")
            return np.nan
        return regr.coef_[0][0]

    def calcBeta_ESTU(self,date:datetime.date):
        if not self.ESTU:
            self.set_ESTU()
        startdate_m1year = datetime.date(date.year-1,date.month,date.day)
        self.load_benchmark(startdate_m1year,date)
        self.load_rf(startdate_m1year,date)
        beta_list = [np.nan]*len(self.ESTU)
        for i,stock in enumerate(self.ESTU):
            beta_list[i] = self.calcBeta_stock(stock,date)
        d = {'Stock':self.ESTU,'Beta':beta_list,'Date':[date]*len(self.ESTU)}
        beta_df = pd.DataFrame(data=d)
        return beta_df
    


if __name__ == '__main__':

    beta = Beta('BETA', None)
    # Check if exceptino works
    #print(beta.calcBeta_stock("KKK",datetime.date(2019,1,1)))

    # Check calculate single stock beta
    #print(beta.calcBeta_stock("AAPL",datetime.date(2019,1,1)))

    # Check calculate ESTU beta
    print(beta.calcBeta_ESTU(datetime.date(2019,1,1)).head())
    
