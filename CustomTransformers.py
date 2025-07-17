from sklearn.base import BaseEstimator, TransformerMixin


class ToDenseTransformer(TransformerMixin):
    def fit(self, X, y=None): return self

    def transform(self, X, y=None):
        return X.toarray() if hasattr(X, "toarray") else X