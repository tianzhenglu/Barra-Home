from typing import List, Dict
import data_loading
import pandas as pd
import standardization as std
import numpy as np
import core_calculation as calc


class Descriptor(object):
    descriptors = None
    """implicitly means descriptor should be a data frame"""
    raw_exposure = None

    def __init__(self, name: str, descriptors_dict: Dict):
        self.name = name
        self.descriptors = descriptors_dict

    def load_raw_data_from_csv_daily(self, file_path: str):
        if self.descriptors is None:
            self.data = pd.read_csv(file_path)

    def load_raw_data_from_csv_weekly(self, file_path: str):
        if self.descriptors is None:
            self.data = pd.read_csv(file_path)

    def load_data_from_db(self):
        pass

    def load_data(self, df: pd.DataFrame):
        """used for testing"""
        self.raw_exposure = df


if __name__ == '__main__':
    data_loading.load_descriptor_daily()
    mid_cap_des = Descriptor('MIDCAP', None)
    ln_cap_des = Descriptor('LNCAP', None)
    mid_cap_des.load_data_from_db('../test_mat')
    ln_cap_des.load_data_from_db('../test_mat')
    weights = pd.DataFrame();
    std_proc = std.STD('cap_weight', weights)
    size_des = Descriptor('size', {'LNCAP': ln_cap_des, 'MIDCAP': mid_cap_des})
    n1 = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    df1 = pd.DataFrame(n1.T, columns=[''])
    print(size_des.descriptors['LNCAP'].name)
