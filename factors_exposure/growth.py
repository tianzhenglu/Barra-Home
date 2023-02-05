from factors_exposure.descriptor import Descriptor
from factors_exposure.factor import Factor
import statsmodels.api as sm
import numpy as np
import pandas as pd


class Growth(Factor):
    name = 'GROWTH'
    descriptors = None

    def calc_weights(self):
        model = sm.OLS(self.data['eps'], self.data.index)

        pass


class SGRO(Descriptor):
    name = 'SGRO'
    data = None

    def get_slope_coefficient(self):
        x = sm.add_constant(self.data['sps'])
        model = sm.OLS(x, self.data.index)
        result = model.fit()
        return result.params[1]


class EGRO(Descriptor):
    name = 'EGRO'
    data = None

    def get_slope_coefficient(self):
        x = sm.add_constant(self.data['eps'])
        model = sm.OLS(x, self.data.index)
        result = model.fit()
        return result.params[1]


class EGIBS(Descriptor):
    name = 'EGIBS'
    data = None


class EGIBS_s(Descriptor):
    name = 'EGIBS_s'
    data = None


if __name__ == '__main__':
    sgro = SGRO('SGRO', None)
    egro = EGRO('EGRO', None)
    egibs = EGIBS('EGIBS', None)
    egibs_s = EGIBS_s('EGIBS_s', None)
    growth_dict = {'SGRO': sgro, 'EGRO': egro, 'EGIBS': egibs, 'EGIBS_s': egibs_s}
    n1 = np.array([[0, 1, 2, 3, 4, 4, 6, 7, 8, 12], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    df1 = pd.DataFrame(n1.T, columns=['eps', 'eps_ret'])
    egro.load_data(df1)
    print(egro.get_slope_coefficient())
    growth_factor = Growth('GROWTH', growth_dict)

