from ..base import _Base


class _ModelBase(_Base):
    def predict(self, X):
        return self.module.predict(X)
