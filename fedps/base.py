from .stats.util import check_FL_type, check_role


class _Base:
    def __init__(self, FL_type: str, role: str, channel=None):
        self.FL_type = check_FL_type(FL_type)
        self.role = check_role(role)
        self.channel = channel

    def check_channel(self):
        if self.channel is None:
            raise ValueError(
                f"For {self.__class__.__name__},"
                f" channel cannot be None in {self.FL_type}FL"
            )

    def Vfit(self, X):
        return self.module.fit(X)

    def Hfit(self, X):
        return self.module.fit(X)

    def fit(self, X=None):
        if self.FL_type == "V":
            return self.Vfit(X)
        else:
            return self.Hfit(X)
