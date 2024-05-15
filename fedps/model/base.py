from ..base import _Base


class _ModelBase(_Base):
    def predict(self, X):
        return self.module.predict(X)

    def fit_predict(self, X):
        return self.fit(X).predict(X)
