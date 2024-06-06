import numpy as np
from fedps.channel import ClientChannel
from fedps.preprocessing import *


channel = ClientChannel(
    local_ip="127.0.0.1",
    local_port=5556,
    remote_ip="127.0.0.1",
    remote_port=5555,
)


""" KBinsDiscretizer """
# X = [[-2, 1, -4, -1],
#      [-1, 2, -3, -0.5]]
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
# y = [1, 2, 6]
# est = LabelBinarizer(
#     FL_type="H",
#     role="client",
#     channel=channel,
# )
# yt = est.fit_transform(y)
# print(yt)


""" LabelEncoder """
# y = [1, 2]
# est = LabelEncoder(
#     FL_type="H",
#     role="client",
#     channel=channel,
# )
# yt = est.fit_transform(y)
# print(yt)


""" MultiLabelBinarizer """
# y = [(1, 2)]
# est = MultiLabelBinarizer(
#     FL_type="H",
#     role="client",
#     channel=channel,
# )
# yt = est.fit_transform(y)
# print(yt)


""" OneHotEncoder """
# X = [["Male", 1], ["Female", 3]]
# est = OneHotEncoder(
#     FL_type="H",
#     role="client",
#     sparse_output=False,
#     channel=channel,
# )
# Xt = est.fit_transform(X)
# print(Xt)


""" OrdinalEncoder """
# X = [["Male", 1], ["Female", 3]]
# est = OrdinalEncoder(
#     FL_type="H",
#     role="client",
#     channel=channel,
# )
# Xt = est.fit_transform(X)
# print(Xt)


""" TargetEncoder """
# X = np.array([["dog"] * 20 + ["cat"] * 30], dtype=object).T
# y = [90.3] * 5 + [80.1] * 15 + [20.4] * 5 + [20.1] * 25
# est = TargetEncoder(
#     FL_type="H",
#     role="client",
#     channel=channel,
# )
# Xt = est.fit_transform(X, y)
# print(Xt)


""" MaxAbsScaler """
# X = [[1., -1., 2.],
#      [2., 0., 0.]]
# est = MaxAbsScaler(
#     FL_type="H",
#     role="client",
#     channel=channel,
# )
# Xt = est.fit_transform(X)
# print(Xt)


""" MinMaxScaler """
X = [[-1, 2], [-0.5, 6]]
est = MinMaxScaler(
    FL_type="H",
    role="client",
    channel=channel,
)
Xt = est.fit_transform(X)
print(Xt)


""" RobustScaler """
# X = [[1., -2., 2.],
#      [-2., 1., 3.]]
# est = RobustScaler(
#     FL_type="H",
#     role="client",
#     channel=channel,
# )
# Xt = est.fit_transform(X)
# print(Xt)


""" StandardScaler """
# X = [[0, 0], [0, 0]]
# est = StandardScaler(
#     FL_type="H",
#     role="client",
#     channel=channel,
# )
# Xt = est.fit_transform(X)
# print(Xt)


""" Normalizer """
# X = [[4, 1], [1, 3], [5, 7]]
# est = Normalizer(
#     FL_type="V",
#     role="client",
#     channel=channel,
# )
# Xt = est.fit_transform(X)
# print(Xt)


""" PowerTransformer """
# X = [[1, 2], [3, 2]]
# est = PowerTransformer(
#     FL_type="H",
#     role="client",
#     channel=channel,
# )
# Xt = est.fit_transform(X)
# print(Xt)


""" QuantileTransformer """
# rng = np.random.RandomState(0)
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
# X = [[0], [1], [2]]
# est = SplineTransformer(
#     FL_type="H",
#     role="client",
#     channel=channel,
# )
# Xt = est.fit_transform(X)
# print(Xt)


""" KNNImputer (H) """
# X = [[1, 2, np.nan], [3, 4, 3]]
# est = KNNImputer(
#     FL_type="H",
#     role="client",
#     n_neighbors=2,
#     channel=channel,
# )
# Xt = est.fit_transform(X)
# print(Xt)


""" KNNImputer (V) """
# X = [[1, 2], [3, 4], [np.nan, 6], [8, 8]]
# est = KNNImputer(
#     FL_type="V",
#     role="client",
#     n_neighbors=2,
#     channel=channel,
# )
# Xt = est.fit_transform(X)
# print(Xt)


""" SimpleImputer """
# X = [[7, 2, 3], [4, np.nan, 6]]
# est = SimpleImputer(
#     FL_type="H",
#     role="client",
#     missing_values=np.nan,
#     strategy="mean",
#     channel=channel,
# )
# Xt = est.fit_transform(X)
# print(Xt)
