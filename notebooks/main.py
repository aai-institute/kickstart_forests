import config 
from typing import Tuple, Optional
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Target Column
COL_LEAF_AREA_INDEX = "lai"

# Input Features
COL_WETNESS = "wetness"

COL_SENTINEL_2A_492 = "Sentinel_2A_492.4"
COL_SENTINEL_2A_559 = "Sentinel_2A_559.8"
COL_SENTINEL_2A_664 = "Sentinel_2A_664.6"
COL_SENTINEL_2A_704 = "Sentinel_2A_704.1"
COL_SENTINEL_2A_740 = "Sentinel_2A_740.5"  
COL_SENTINEL_2A_782 = "Sentinel_2A_782.8"
COL_SENTINEL_2A_832 = "Sentinel_2A_832.8"
COL_SENTINEL_2A_864 = "Sentinel_2A_864.7"
COL_SENTINEL_2A_1613 = "Sentinel_2A_1613.7"
COL_SENTINEL_2A_2202 = "Sentinel_2A_2202.4"

COL_SENTINEL_VALUES = [COL_SENTINEL_2A_492, COL_SENTINEL_2A_559, COL_SENTINEL_2A_664, COL_SENTINEL_2A_704, COL_SENTINEL_2A_740,
    COL_SENTINEL_2A_782, COL_SENTINEL_2A_832,COL_SENTINEL_2A_864,COL_SENTINEL_2A_1613,COL_SENTINEL_2A_2202]

class Dataset:
    def __init__(self, num_samples: Optional[int] = None, drop_na: bool = True ,drop_tree_species : bool = True,
                  random_seed: int = 42):
        """
        :param num_samples: the number of samples to draw from the data frame; if None, use all samples
        :param drop_na: whether to drop null values or not
        :param drop_tree_species: whether to drop tree species or not
        :param random_seed: the random seed to use when sampling data points
        """
        self.num_samples = num_samples
        self.drop_na = drop_na
        self.drop_tree_species = drop_tree_species
        self.random_seed = random_seed

    def load_data_frame(self) -> pd.DataFrame:
        """
        :return: the full data frame for this dataset (including the class column)
        """

        if self.drop_na:
            df = pd.read_csv(config.csv_data_path(), index_col= 0).dropna()
        else:
            df = pd.read_csv(config.csv_data_path(), index_col= 0)

        if self.drop_tree_species:
            df.drop('treeSpecies', axis = 1, inplace= True)

        if self.num_samples is not None:
            df = df.sample(self.num_samples, random_state=self.random_seed)
        return df

    def load_xy(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        :return: a pair (X, y) where X is the data frame containing all attributes and y is the corresping series of class values
        """
        df = self.load_data_frame()
        return df.drop(columns=COL_LEAF_AREA_INDEX), df[COL_LEAF_AREA_INDEX]

if __name__ == '__main__':

    dataset = Dataset()
    X, y = dataset.load_xy()

    # project to columns used by models
    cols_used_by_models = [COL_WETNESS, *COL_SENTINEL_VALUES]
    X = X[cols_used_by_models]

    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42)

    m = LinearRegression()

    m.fit(X_train, y_train)

    m.score(X_train, y_train)

    lin_reg = m.score(X_test, y_test)

    print(f"Linear Regression Score:{lin_reg}")

    rf = RandomForestRegressor()

    rf.fit(X_train, y_train)

    rf.score(X_train, y_train)
    
    rf_score = rf.score(X_test, y_test)

    print(f"Random Forest Score:{rf_score}")

