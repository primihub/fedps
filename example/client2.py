import numpy as np
from fedps.channel import ClientChannel
from fedps.preprocessing import *


channel = ClientChannel(
    local_ip="127.0.0.1",
    local_port=5557,
    remote_ip="127.0.0.1",
    remote_port=5555,
)


""" KBinsDiscretizer """
# X = [[0, 3, -2, 0.5],
#      [1, 4, -1, 2]]
# est = KBinsDiscretizer(
#     FL_type="H",
#     role="client",
#     n_bins=3,
#     encode="ordinal",
#     strategy="uniform",
#     channel=channel
# )
# Xt = est.fit_transform(X)
# print(Xt)


""" LabelBinarizer """
# y = [4, 2]
# est = LabelBinarizer(
#     FL_type="H",
#     role="client",
#     channel=channel,
# )
# yt = est.fit_transform(y)
# print(yt)


""" LabelEncoder """
# y = [2, 6]
# est = LabelEncoder(
#     FL_type="H",
#     role="client",
#     channel=channel,
# )
# yt = est.fit_transform(y)
# print(yt)


""" MultiLabelBinarizer """
# y = [(3,)]
# est = MultiLabelBinarizer(
#     FL_type="H",
#     role="client",
#     channel=channel,
# )
# yt = est.fit_transform(y)
# print(yt)


""" OneHotEncoder """
# X = [["Female", 2]]
# est = OneHotEncoder(
#     FL_type="H",
#     role="client",
#     sparse_output=False,
#     channel=channel,
# )
# Xt = est.fit_transform(X)
# print(Xt)


""" OrdinalEncoder """
# X = [["Female", 2]]
# est = OrdinalEncoder(
#     FL_type="H",
#     role="client",
#     channel=channel,
# )
# Xt = est.fit_transform(X)
# print(Xt)


""" TargetEncoder """
# X = np.array([["snake"] * 38], dtype=object).T
# y = [21.2] * 8 + [49] * 30
# est = TargetEncoder(
#     FL_type="H",
#     role="client",
#     channel=channel,
# )
# Xt = est.fit_transform(X, y)
# print(Xt)


""" MaxAbsScaler """
# X = [[0., 1., -1.]]
# est = MaxAbsScaler(
#     FL_type="H",
#     role="client",
#     channel=channel,
# )
# Xt = est.fit_transform(X)
# print(Xt)


""" MinMaxScaler """
X = [[0, 10], [1, 18]]
est = MinMaxScaler(
    FL_type="H",
    role="client",
    channel=channel,
)
Xt = est.fit_transform(X)
print(Xt)


""" RobustScaler """
# X = [[4., 1., -2.]]
# est = RobustScaler(
#     FL_type="H",
#     role="client",
#     channel=channel,
# )
# Xt = est.fit_transform(X)
# print(Xt)


""" StandardScaler """
# X = [[1, 1], [1, 1]]
# est = StandardScaler(
#     FL_type="H",
#     role="client",
#     channel=channel,
# )
# Xt = est.fit_transform(X)
# print(Xt)


""" Normalizer """
# X = [[2, 2], [9, 3], [5, 1]]
# est = Normalizer(
#     FL_type="V",
#     role="client",
#     channel=channel,
# )
# Xt = est.fit_transform(X)
# print(Xt)


""" PowerTransformer """
# X = [[4, 5]]
# est = PowerTransformer(
#     FL_type="H",
#     role="client",
#     channel=channel,
# )
# Xt = est.fit_transform(X)
# print(Xt)


""" QuantileTransformer """
# rng = np.random.RandomState(1)
# X = rng.normal(loc=0.5, scale=0.25, size=(25, 1))
# est = QuantileTransformer(
#     FL_type="H",
#     role="client",
#     n_quantiles=10,
#     channel=channel,
# )
# Xt = est.fit_transform(X)
# print(Xt)


""" SplineTransformer """
# X = [[3], [4], [5]]
# est = SplineTransformer(
#     FL_type="H",
#     role="client",
#     channel=channel,
# )
# Xt = est.fit_transform(X)
# print(Xt)


""" SimpleImputer """
# X = [[10, 5, np.nan]]
# est = SimpleImputer(
#     FL_type="H",
#     role="client",
#     missing_values=np.nan,
#     strategy="mean",
#     channel=channel,
# )
# Xt = est.fit_transform(X)
# print(Xt)
