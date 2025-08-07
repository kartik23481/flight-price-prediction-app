import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel

class RBFPercentileSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None, percentiles=[0.25, 0.5, 0.75], gamma=0.1):
        self.variables = variables
        self.percentiles = percentiles
        self.gamma = gamma

    def fit(self, df, y=None):
        if not self.variables:
            self.variables = df.select_dtypes(include="number").columns.to_list()
        self.reference_values_ = {
            col: (
                df.loc[:, col]
                .quantile(self.percentiles)
                .values
                .reshape(-1, 1)
            )
            for col in self.variables
        }
        return self

    def transform(self, df):
        # Convert to DataFrame if X is ndarray
        if isinstance(df, np.ndarray):
            df = pd.DataFrame(df, columns=self.variables)
        
        objects = []
        for col in self.variables:
            columns = [f"{col}_rbf_{int(percentile * 100)}" for percentile in self.percentiles]
            obj = pd.DataFrame(
                data=rbf_kernel(df[[col]], Y=self.reference_values_[col], gamma=self.gamma),
                columns=columns,
                index=df.index
            )
            objects.append(obj)
        return pd.concat(objects, axis=1)

