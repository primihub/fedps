import numpy as np
from fedps.channel import ServerChannel
from fedps.preprocessing import *


channel = ServerChannel(
    local_ip="127.0.0.1",
    local_port=5555,
    remote_ip=["127.0.0.1", "127.0.0.1"],
    remote_port=[5556, 5557],
)


""" KBinsDiscretizer """
# est = KBinsDiscretizer(
#     n_bins=3,
#     encode="ordinal",
#     strategy="uniform",
#     FL_type="H",
#     role="server",
#     channel=channel,
# )
# est.fit()


""" LabelBinarizer """
# est = LabelBinarizer(
#     FL_type="H",
#     role="server",
#     channel=channel,
# )
# est.fit()


""" LabelEncoder """
# est = LabelEncoder(
#     FL_type="H",
#     role="server",
#     channel=channel,
# )
# est.fit()


""" MultiLabelBinarizer """
# est = MultiLabelBinarizer(
#     FL_type="H",
#     role="server",
#     channel=channel,
# )
# est.fit()


""" OneHotEncoder """
# est = OneHotEncoder(
#     sparse_output=False,
#     FL_type="H",
#     role="server",
#     channel=channel,
# )
# est.fit()


""" OrdinalEncoder """
# est = OneHotEncoder(
#     FL_type="H",
#     role="server",
#     channel=channel,
# )
# est.fit()


""" TargetEncoder """
# est = TargetEncoder(
#     FL_type="H",
#     role="server",
#     channel=channel,
# )
# est.fit_transform()


""" MaxAbsScaler """
# est = MaxAbsScaler(
#     FL_type="H",
#     role="server",
#     channel=channel,
# )
# est.fit()


""" MinMaxScaler """
est = MinMaxScaler(
    FL_type="H",
    role="server",
    channel=channel,
)
est.fit()


""" RobustScaler """
# est = RobustScaler(
#     FL_type="H",
#     role="server",
#     channel=channel,
# )
# est.fit()


""" StandardScaler """
# est = StandardScaler(
#     FL_type="H",
#     role="server",
#     channel=channel,
# )
# est.fit()


""" Normalizer """
# X = [[5, 7, 5, 1]]
# est = Normalizer(
#     FL_type="V",
#     role="host",
#     channel=channel,
# )
# Xt = est.fit_transform(X)
# print(Xt)


""" PowerTransformer """
# est = PowerTransformer(
#     FL_type="H",
#     role="server",
#     channel=channel,
# )
# est.fit()


""" QuantileTransformer """
# est = QuantileTransformer(
#     n_quantiles=10,
#     FL_type="H",
#     role="server",
#     channel=channel,
# )
# est.fit()


""" SplineTransformer """
# est = SplineTransformer(
#     FL_type="H",
#     role="server",
#     channel=channel,
# )
# est.fit()


""" SimpleImputer """
# est = SimpleImputer(
#     missing_values=np.nan,
#     strategy="mean",
#     FL_type="H",
#     role="server",
#     channel=channel,
# )
# est.fit()
