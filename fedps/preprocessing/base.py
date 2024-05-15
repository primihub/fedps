from ..base import _Base


class _PreprocessBase(_Base):
    def transform(self, X):
        return self.module.transform(X)

    def fit_transform(self, X):
        return self.fit(X).transform(X)
