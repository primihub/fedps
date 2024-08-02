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
#     FL_type="H",
#     role="server",
#     n_bins=3,
#     encode="ordinal",
#     strategy="uniform",
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
#     FL_type="H",
#     role="server",
#     sparse_output=False,
#     channel=channel,
# )
# est.fit()


""" OrdinalEncoder """
# est = OrdinalEncoder(
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
# est = Normalizer(
#     FL_type="V",
#     role="server",
#     channel=channel,
# )
# est.transform()


""" PowerTransformer """
# est = PowerTransformer(
#     FL_type="H",
#     role="server",
#     channel=channel,
# )
# est.fit()


""" QuantileTransformer """
# est = QuantileTransformer(
#     FL_type="H",
#     role="server",
#     n_quantiles=10,
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


""" IterativeImputer (H) """
# est = IterativeImputer(
#     FL_type="H",
#     role="server",
#     channel=channel,
# )
# est.fit()


""" IterativeImputer (V) """
# est = IterativeImputer(
#     FL_type="V",
#     role="server",
#     channel=channel,
# )
# est.fit_transform()


""" KNNImputer (H) """
# est = KNNImputer(
#     FL_type="H",
#     role="server",
#     n_neighbors=2,
#     channel=channel,
# )
# est.fit_transform()


""" KNNImputer (V) """
# est = KNNImputer(
#     FL_type="V",
#     role="server",
#     n_neighbors=2,
#     channel=channel,
# )
# est.fit_transform()


""" SimpleImputer """
# est = SimpleImputer(
#     FL_type="H",
#     role="server",
#     missing_values=np.nan,
#     strategy="mean",
#     channel=channel,
# )
# est.fit()
