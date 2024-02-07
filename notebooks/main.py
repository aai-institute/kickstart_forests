import re
from typing import Tuple, Optional

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.manifold import TSNE

import config

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

COL_SENTINEL_VALUES = [
    COL_SENTINEL_2A_492,
    COL_SENTINEL_2A_559,
    COL_SENTINEL_2A_664,
    COL_SENTINEL_2A_704,
    COL_SENTINEL_2A_740,
    COL_SENTINEL_2A_782,
    COL_SENTINEL_2A_832,
    COL_SENTINEL_2A_864,
    COL_SENTINEL_2A_1613,
    COL_SENTINEL_2A_2202,
]


class Dataset:
    def __init__(
        self,
        num_samples: Optional[int] = None,
        drop_na: bool = True,
        drop_tree_species: bool = True,
        random_seed: int = 42,
    ):
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
            df = pd.read_csv(config.csv_data_path(), index_col=0).dropna()
        else:
            df = pd.read_csv(config.csv_data_path(), index_col=0)

        if self.drop_tree_species:
            df.drop("treeSpecies", axis=1, inplace=True)

        if self.num_samples is not None:
            df = df.sample(self.num_samples, random_state=self.random_seed)

        column_names = list(df.columns)
        self.wavelength_columns = [x for x in column_names if x.startswith("w")][1:]
        self.sentinel_columns = [x for x in column_names if x.startswith("Sentinel")]
        return df

    def load_xy(self, use_tsne: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
        """
        :return: a pair (X, y) where X is the data frame containing all attributes and y is the corresping series of class values
        """
        df = self.load_data_frame()
        if use_tsne:
            tsne_model = TSNE(
                n_components=3,
                learning_rate="auto",
                init="random",
                perplexity=30,
                random_state=0,
            )
            wavelength_feature_data = df.loc[:, self.wavelength_columns]
            wavelength_embeddings = tsne_model.fit_transform(wavelength_feature_data)

            sentinel_feature_data = df.loc[:, self.sentinel_columns]
            tsne_model.set_params(n_components=3, perplexity=5)
            sentinel_embeddings = tsne_model.fit_transform(sentinel_feature_data)

            embeddings = np.concatenate(
                (sentinel_embeddings, wavelength_embeddings), axis=1
            )

            wetness_features = df.loc[:, COL_WETNESS].values
            wetness_features = np.expand_dims(wetness_features, axis=1)
            embeddings = np.concatenate((embeddings, wetness_features), axis=1)
            return embeddings, np.expand_dims(df[COL_LEAF_AREA_INDEX].values, axis=1)
        else:
            return df.drop(columns=COL_LEAF_AREA_INDEX), df[COL_LEAF_AREA_INDEX]


class ModelFactory:
    COLS_USED_BY_ORIGINAL_MODELS = [COL_WETNESS, *COL_SENTINEL_VALUES]

    @classmethod
    def create_logistic_regression_orig(cls):
        return Pipeline(
            [
                (
                    "project_scale",
                    ColumnTransformer(
                        [("scaler", StandardScaler(), cls.COLS_USED_BY_ORIGINAL_MODELS)]
                    ),
                ),
                ("model", linear_model.LinearRegression()),
            ]
        )

    @classmethod
    def create_random_forest_orig(cls):
        return Pipeline(
            [
                (
                    "project_scale",
                    ColumnTransformer(
                        [("scaler", StandardScaler(), cls.COLS_USED_BY_ORIGINAL_MODELS)]
                    ),
                ),
                ("model", RandomForestRegressor(n_estimators=100)),
            ]
        )

    @classmethod
    def pure_random_forest(cls):
        return RandomForestRegressor(n_estimators=100, max_depth=100, random_state=42)

    @classmethod
    def pure_mlp(cls):
        return MLPRegressor(
            hidden_layer_sizes=(1000, 100, 10),
            max_iter=1000,
            random_state=42,
            early_stopping=True,
        )


if __name__ == "__main__":
    use_tsne = True

    dataset = Dataset()
    X, y = dataset.load_xy(use_tsne=use_tsne)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    models = (
        [
            ModelFactory.create_logistic_regression_orig(),
            ModelFactory.create_random_forest_orig(),
        ]
        if not use_tsne
        else [ModelFactory.pure_random_forest(), ModelFactory.pure_mlp()]
    )

    # evaluate models
    for model in models:
        print(f"Evaluating model:\n{model}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"R2_score: {r2_score(y_test, y_pred)}")
        print(f"MSE: {mean_squared_error(y_test, y_pred)}")
