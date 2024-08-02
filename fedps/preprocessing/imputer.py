import numpy as np
from typing import Callable
from sklearn.impute import KNNImputer as SKL_KNNImputer
from sklearn.impute import SimpleImputer as SKL_SimpleImputer
from sklearn.metrics import pairwise_distances_chunked
from sklearn.neighbors._base import _get_weights
from sklearn.utils._encode import _unique
from sklearn.utils._mask import _get_mask
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted
from .base import _PreprocessBase
from .util import validate_quantile_sketch_params
from ..sketch import (
    send_local_quantile_sketch,
    merge_local_quantile_sketch,
    get_quantiles,
    send_local_fi_sketch,
    merge_local_fi_sketch,
    get_frequent_items,
)
from ..util import import_is_scalar_nan

is_scalar_nan = import_is_scalar_nan()


class KNNImputer(_PreprocessBase):
    def __init__(
        self,
        FL_type: str,
        role: str,
        missing_values=np.nan,
        n_neighbors=5,
        weights="uniform",
        copy=True,
        add_indicator=False,
        keep_empty_features=False,
        channel=None,
    ):
        super().__init__(FL_type, role, channel)
        self.check_channel()
        self.module = SKL_KNNImputer(
            missing_values=missing_values,
            n_neighbors=n_neighbors,
            weights=weights,
            copy=copy,
            add_indicator=add_indicator,
            keep_empty_features=keep_empty_features,
        )

    def fit(self, X=None):
        if self.role == "client":
            self.module.fit(X)
            masked_X = np.ma.masked_array(
                self.module._fit_X, mask=self.module._mask_fit_X
            )
        elif self.role == "server":
            self.module._validate_params()

        if self.FL_type == "H":
            if self.role == "client":
                self.channel.send("valid_mask", self.module._valid_mask)
                self.module._valid_mask = self.channel.recv("valid_mask")

                col_sum = np.ma.sum(masked_X, axis=0).data
                self.channel.send("col_sum", col_sum)
                col_n_non_missing = self.module._fit_X.shape[0] - np.sum(
                    self.module._mask_fit_X, axis=0
                )
                if np.ptp(col_n_non_missing) == 0:
                    col_n_non_missing = col_n_non_missing[0]
                self.channel.send("col_n_non_missing", col_n_non_missing)

            elif self.role == "server":
                valid_mask = self.channel.recv_all("valid_mask")
                valid_mask = np.any(valid_mask, axis=0)
                self.channel.send_all("valid_mask", valid_mask)
                self.module._valid_mask = valid_mask
                self.module.n_features_in_ = valid_mask.size

                col_sum = self.channel.recv_all("col_sum")
                col_sum = np.sum(col_sum, axis=0)
                col_n_non_missing = self.channel.recv_all("col_n_non_missing")
                col_n_non_missing = sum(col_n_non_missing)
                self._mean = col_sum / col_n_non_missing
                if self.module.keep_empty_features:
                    self._mean[self.module._valid_mask] = np.nan

        else:  # self.FL_type == "V"
            if self.role == "client":
                self._mean = np.ma.mean(masked_X, axis=0).data
                if self.module.keep_empty_features:
                    self._mean[self.module._valid_mask] = np.nan

        return self

    def fit_transform(self, X=None):
        return self.fit(X).transform(X)

    def transform(self, X=None):
        if self.FL_type == "V":
            return self.Vtrasform(X)
        else:
            return self.Htrasform(X)

    def Htrasform(self, X):
        check_is_fitted(self.module)
        if not is_scalar_nan(self.module.missing_values):
            force_all_finite = True
        else:
            force_all_finite = "allow-nan"

        valid_mask = self.module._valid_mask

        if self.role == "client":
            if X is None:
                self.channel.send("X_nan", None)
            else:
                X = self.module._validate_data(
                    X,
                    accept_sparse=False,
                    dtype=FLOAT_DTYPES,
                    force_all_finite=force_all_finite,
                    copy=self.module.copy,
                    reset=False,
                )

                mask = _get_mask(X, self.module.missing_values)
                X_indicator = self.module._transform_indicator(mask)

                has_missing = np.any(mask[:, valid_mask])
                if not has_missing:
                    # No missing values in X
                    self.channel.send("X_nan", None)
                    # Removes columns where the training data is all nan
                    if self.module.keep_empty_features:
                        Xc = X
                        Xc[:, ~valid_mask] = 0
                    else:
                        Xc = X[:, valid_mask]
                        Xt = self.module._concatenate_indicator(Xc, X_indicator)
                else:
                    # Send local data with missing values to server
                    row_missing_idx = np.flatnonzero(mask[:, valid_mask].any(axis=1))
                    self.channel.send("X_nan", X[row_missing_idx, :])

            X_nan = self.channel.recv("X_nan")
            if X_nan is None:
                return Xt if X is not None else None

            mask_nan = _get_mask(X_nan, self.module.missing_values)

            # All samples of received data have missing values
            row_idx = np.arange(X_nan.shape[0])

            col_missing_idx = np.arange(X_nan.shape[1])
            # column has missing values must be valid
            col_missing_idx = col_missing_idx[mask_nan.any(axis=0) * valid_mask]

            non_missing_fix_X = np.logical_not(self.module._mask_fit_X)

            # local top k minimum distance and corresponding value
            min_k_dist = [[] for _ in range(col_missing_idx.size)]
            min_k_value = [[] for _ in range(col_missing_idx.size)]

            def process_chunk(dist_chunk, start):
                row_idx_chunk = row_idx[start : start + len(dist_chunk)]

                # Find min_k_dist and min_k_value by column
                for col_idx, col in enumerate(col_missing_idx):
                    col_mask = mask_nan[row_idx_chunk, col]

                    potential_donors_idx = np.flatnonzero(non_missing_fix_X[:, col])
                    potential_donors_values = self.module._fit_X[
                        potential_donors_idx, col
                    ]

                    # receivers_idx are indices in X_nan
                    receivers_idx = row_idx_chunk[np.flatnonzero(col_mask)]

                    # distances for samples that needed imputation for column
                    dist_subset = dist_chunk[row_idx[receivers_idx] - start][
                        :, potential_donors_idx
                    ]

                    n_neighbors = min(
                        self.module.n_neighbors, len(potential_donors_idx)
                    )

                    # Get donors of local top k minimum distance
                    donors_idx = np.argpartition(dist_subset, n_neighbors - 1, axis=1)[
                        :, :n_neighbors
                    ]

                    # Get distance from donors
                    donors_dist = dist_subset[
                        np.arange(donors_idx.shape[0])[:, None], donors_idx
                    ]

                    # Remove nans then store distance and value by column
                    for i, dist in enumerate(donors_dist):
                        non_nan_mask = ~np.isnan(dist)
                        min_k_dist[col_idx].append(dist[non_nan_mask])
                        donors_values = potential_donors_values.take(donors_idx[i])
                        min_k_value[col_idx].append(donors_values[non_nan_mask])

            gen = pairwise_distances_chunked(
                X_nan,
                self.module._fit_X,
                metric=self.module.metric,
                missing_values=self.module.missing_values,
                force_all_finite=force_all_finite,
                reduce_func=process_chunk,
            )
            for chunk in gen:
                # process_chunk store min_k_dist and min_k_value. No return value.
                pass

            self.channel.send("min_k_dist", min_k_dist)
            min_k_idx = self.channel.recv("min_k_idx")

            if len(min_k_idx) > 0:
                donor_value = []
                # select value according to index
                for col_idx, row_idx, val_idx in min_k_idx:
                    val = min_k_value[col_idx][row_idx][val_idx]

                    # compute local sum of value
                    if self.module.weights == "uniform":
                        donor_value.append(val.sum())
                    else:  # "distance"
                        dist = min_k_dist[col_idx][row_idx][val_idx]
                        # weight is the reciprocal of distance
                        if np.any(dist == 0):
                            # zero distance has weight one
                            donor_value.append(val.sum())
                        else:
                            donor_value.append((val / dist).sum())
                self.channel.send("donor_value", donor_value)

            if X is None:
                return

            if has_missing:
                impute_value = self.channel.recv("impute_value")
                X[np.nonzero(mask[:, valid_mask])] = impute_value

            # Removes columns where the training data is all nan
            if self.module.keep_empty_features:
                Xc = X
                Xc[:, ~valid_mask] = 0
            else:
                Xc = X[:, valid_mask]
            return self.module._concatenate_indicator(Xc, X_indicator)

        elif self.role == "server":
            n_client = self.channel.n_client
            X_nan = self.channel.recv_all("X_nan")
            client_n_samples = [0 if x is None else x.shape[0] for x in X_nan]

            # Remove None from client data (indicate no missing values)
            X_nan = [x for x in X_nan if x is not None]
            if len(X_nan) == 0:
                # No missing values
                self.channel.send_all("X_nan", None)
                return

            X_nan = np.vstack(X_nan)
            self.channel.send_all("X_nan", X_nan)

            mask_nan = _get_mask(X_nan, self.module.missing_values)
            col_missing_idx = np.arange(X_nan.shape[1])
            # column has missing values must be valid
            col_missing_idx = col_missing_idx[mask_nan.any(axis=0) * valid_mask]

            # Only store columns which have missing values and valid
            X_nan = X_nan[:, col_missing_idx]
            mask_nan = mask_nan[:, col_missing_idx]

            # the number of missing values per column
            col_n_missing = mask_nan.sum(axis=0)
            # mean = value_sum / weights for each column
            value_sum = [np.zeros(n) for n in col_n_missing]
            weights = [np.ones(n) for n in col_n_missing]

            min_k_dist = self.channel.recv_all("min_k_dist")
            min_k_idx = [[] for _ in range(n_client)]

            for col_idx, col in enumerate(col_missing_idx):
                for row_idx in range(col_n_missing[col_idx]):

                    dist = np.array([], dtype=float)
                    client_idx = np.array([], dtype=int)
                    value_idx = np.array([], dtype=int)

                    # Merge distance from all clients by each position
                    for ci, client_min_k_dist in enumerate(min_k_dist):
                        client_dist = client_min_k_dist[col_idx][row_idx]
                        size = client_dist.size
                        if size > 0:
                            dist = np.r_[dist, client_dist]
                            # Set client_idx and value_idx for each distance
                            client_idx = np.r_[client_idx, [ci] * size]
                            value_idx = np.r_[value_idx, np.arange(size)]

                    dist_size = dist.size
                    if dist_size == 0:
                        # Impute with mean if no distance received
                        # the default weight is one
                        value_sum[col_idx][row_idx] = self._mean[col]
                        continue

                    n_neighbors = min(self.module.n_neighbors, dist_size)
                    donors_idx = np.argpartition(dist, n_neighbors - 1)[:n_neighbors]

                    # Select global top-k minimum distance
                    dist = dist[donors_idx]
                    client_idx = client_idx[donors_idx]
                    value_idx = value_idx[donors_idx]

                    # Set weights for each position of missing values
                    if self.module.weights == "uniform":
                        weights[col_idx][row_idx] = n_neighbors
                    else:  # "distance"
                        with np.errstate(divide="ignore"):
                            weight = 1.0 / dist
                        # check if any zero distance exist
                        inf_mask = np.isinf(weight)
                        if np.any(inf_mask):
                            # set infinite weight (zero distance) to one
                            # set others weight (non-zero distance) to zero
                            weight = np.ones(inf_mask.sum())
                            # remove idx with non-zero distance
                            client_idx = client_idx[inf_mask]
                            value_idx = value_idx[inf_mask]
                        weights[col_idx][row_idx] = weight.sum()

                    # Group value_idx by client_idx
                    client_value_idx = {ci: [] for ci in np.unique(client_idx)}
                    for ci, vi in zip(client_idx, value_idx):
                        client_value_idx[ci].append(vi)

                    # value idx for each client is a sparse matirx
                    # therefore also include indices of column and row
                    for ci, vi in client_value_idx.items():
                        min_k_idx[ci].append((col_idx, row_idx, vi))

            # Each client receives a different min_k_idx
            self.channel.send_all_diff("min_k_idx", min_k_idx)

            # Receive values which correspond to top k min distance
            # Only reveive from clients which min_k_idx is not empty
            client_mask = [len(idx) > 0 for idx in min_k_idx]
            client_idx = np.arange(n_client)[client_mask]
            donor_value = self.channel.recv_selected("donor_value", client_idx)

            # Compute sum of values then divide by their weights
            for ci, value in zip(client_idx, donor_value):
                for (col_idx, row_idx, _), val in zip(min_k_idx[ci], value):
                    value_sum[col_idx][row_idx] += val

            for col_idx in range(X_nan.shape[1]):
                X_nan[:, col_idx][mask_nan[:, col_idx]] = (
                    value_sum[col_idx] / weights[col_idx]
                )

            # Send imputed value to each client
            impute_value = []
            start = end = 0
            for n in client_n_samples:
                if n == 0:
                    continue
                start = end
                end += n
                impute_value.append(X_nan[start:end][mask_nan[start:end]])
            # Only send to clients which data contain missing values
            client_mask = [n > 0 for n in client_n_samples]
            client_idx = np.arange(n_client)[client_mask]
            self.channel.send_selected_diff("impute_value", impute_value, client_idx)

    def Vtrasform(self, X):
        if not is_scalar_nan(self.module.missing_values):
            force_all_finite = True
        else:
            force_all_finite = "allow-nan"

        if self.role == "client":
            check_is_fitted(self.module)
            X = self.module._validate_data(
                X,
                accept_sparse=False,
                dtype=FLOAT_DTYPES,
                force_all_finite=force_all_finite,
                copy=self.module.copy,
                reset=False,
            )
            n_samples, n_features = X.shape
            self.channel.send("n_samples", n_samples)
            self.channel.send("n_features", n_features)

            valid_mask = self.module._valid_mask
            mask = _get_mask(X, self.module.missing_values)
            X_indicator = self.module._transform_indicator(mask)

            has_missing = np.any(mask[:, valid_mask])
            if not has_missing:
                # Removes columns where the training data is all nan
                if self.module.keep_empty_features:
                    Xc = X
                    Xc[:, ~valid_mask] = 0
                else:
                    Xc = X[:, valid_mask]
                    Xt = self.module._concatenate_indicator(Xc, X_indicator)

            # rows contain missing values for client
            client_row_missing_idx = np.flatnonzero(mask[:, valid_mask].any(axis=1))
            self.channel.send("client_row_missing_idx", client_row_missing_idx)
            row_missing_idx = self.channel.recv("row_missing_idx")

            if row_missing_idx.size == 0:
                return Xt

            non_missing_fix_X = np.logical_not(self.module._mask_fit_X)

            dist_client, non_missing_count = [], []

            def process_chunk(dist_chunk, start):
                row_missing_chunk = row_missing_idx[start : start + len(dist_chunk)]
                # count the number of features which is not a missing value
                # for both test data and train data
                non_missing_chunk = 1 - mask[row_missing_chunk, :]
                non_missing_count_chunk = np.dot(non_missing_chunk, non_missing_fix_X.T)
                non_missing_count.append(non_missing_count_chunk)

                # modify the nan_euclidean_distances to
                # unscaled squared euclidean distances
                dist_chunk[np.isnan(dist_chunk)] = 0.0
                np.square(dist_chunk, out=dist_chunk)
                dist_chunk /= n_features
                dist_chunk *= non_missing_count_chunk
                return dist_chunk

            gen = pairwise_distances_chunked(
                X[row_missing_idx, :],
                self.module._fit_X,
                metric=self.module.metric,
                missing_values=self.module.missing_values,
                force_all_finite=force_all_finite,
                reduce_func=process_chunk,
            )
            for chunk in gen:
                dist_client.append(chunk)
                # process_chunk return dist_chunk and store non_missing_count.
                pass

            dist_client = np.vstack(dist_client)
            non_missing_count = np.vstack(non_missing_count)
            self.channel.send("dist_client", dist_client)
            self.channel.send("non_missing_count", non_missing_count)

            if has_missing:
                # Maps from indices from X to indices in dist matrix
                dist_idx_map = np.zeros(n_samples, dtype=int)
                dist_idx_map[client_row_missing_idx] = np.arange(
                    client_row_missing_idx.size
                )

                col_missing_idx = np.arange(n_features)
                # column has missing values must be valid
                col_missing_idx = col_missing_idx[mask.any(axis=0) * valid_mask]

                dist_server = self.channel.recv("dist_server")

                for col in col_missing_idx:
                    col_mask = mask[:, col]
                    # receivers_idx are indices in X
                    receivers_idx = np.flatnonzero(col_mask)
                    potential_donors_idx = np.flatnonzero(non_missing_fix_X[:, col])

                    # distances for samples that needed imputation for column
                    dist_subset = dist_server[dist_idx_map[receivers_idx]][
                        :, potential_donors_idx
                    ]

                    # receivers with all nan distances impute with mean
                    all_nan_dist_mask = np.isnan(dist_subset).all(axis=1)
                    all_nan_receivers_idx = receivers_idx[all_nan_dist_mask]

                    if all_nan_receivers_idx.size:
                        X[all_nan_receivers_idx, col] = self._mean[col]

                        if len(all_nan_receivers_idx) == len(receivers_idx):
                            # all receivers imputed with mean
                            continue

                        # receivers with at least one defined distance
                        receivers_idx = receivers_idx[~all_nan_dist_mask]
                        dist_subset = dist_server[dist_idx_map[receivers_idx]][
                            :, potential_donors_idx
                        ]

                    n_neighbors = min(
                        self.module.n_neighbors, len(potential_donors_idx)
                    )

                    # Get donors
                    donors_idx = np.argpartition(dist_subset, n_neighbors - 1, axis=1)[
                        :, :n_neighbors
                    ]

                    # Get weight matrix from distance matrix
                    donors_dist = dist_subset[
                        np.arange(donors_idx.shape[0])[:, None], donors_idx
                    ]

                    weight_matrix = _get_weights(donors_dist, self.module.weights)

                    # fill nans with zeros
                    if weight_matrix is not None:
                        weight_matrix[np.isnan(weight_matrix)] = 0.0
                    else:
                        weight_matrix = np.ones_like(donors_dist)
                        weight_matrix[np.isnan(donors_dist)] = 0.0

                    donors = self.module._fit_X[potential_donors_idx, col].take(
                        donors_idx
                    )

                    value = np.average(donors, axis=1, weights=weight_matrix)
                    X[receivers_idx, col] = value

            # Removes columns where the training data is all nan
            if self.module.keep_empty_features:
                Xc = X
                Xc[:, ~valid_mask] = 0
            else:
                Xc = X[:, valid_mask]
            return self.module._concatenate_indicator(Xc, X_indicator)

        elif self.role == "server":
            client_n_samples = self.channel.recv_all("n_samples")
            if np.ptp(client_n_samples) != 0:
                raise RuntimeError(
                    "The number of samples are not equal for"
                    f" all clients: {client_n_samples}"
                )
            n_samples = client_n_samples[0]

            client_n_features = self.channel.recv_all("n_features")
            n_features = sum(client_n_features)
            self.module.n_features_in_ = n_features

            client_row_missing_idx = self.channel.recv_all("client_row_missing_idx")
            # All rows contain missing values
            row_missing_idx = np.unique(np.concatenate(client_row_missing_idx))
            self.channel.send_all("row_missing_idx", row_missing_idx)

            if row_missing_idx.size == 0:
                return

            dist_server = self.channel.recv_all("dist_client")
            non_missing_count = self.channel.recv_all("non_missing_count")

            dist_server = np.sum(dist_server, axis=0)
            non_missing_count = np.sum(non_missing_count, axis=0)

            dist_server[non_missing_count == 0] = np.nan
            # avoid divide by zero
            np.maximum(1, non_missing_count, out=non_missing_count)
            dist_server /= non_missing_count

            # For uniform weights, multiply by n_features and
            # take the square root are not needed
            # if self.module.weights == "distance":
            dist_server *= n_features
            np.sqrt(dist_server, out=dist_server)

            # Maps from indices from X to indices in dist matrix
            dist_idx_map = np.zeros(n_samples, dtype=int)
            dist_idx_map[row_missing_idx] = np.arange(row_missing_idx.size)

            dist_select = []
            client_idx = []
            for ci, missing_idx in enumerate(client_row_missing_idx):
                if missing_idx.size > 0:
                    client_idx.append(ci)
                    dist_select.append(dist_server[dist_idx_map[missing_idx], :])
            # Send distance selected by row_missing_idx for each client
            # Only send to clients which data contain missing values
            self.channel.send_selected_diff("dist_server", dist_select, client_idx)


class SimpleImputer(_PreprocessBase):
    def __init__(
        self,
        FL_type: str,
        role: str,
        missing_values=np.nan,
        strategy="mean",
        fill_value=None,
        copy=True,
        add_indicator=False,
        keep_empty_features=False,
        sketch_name="KLL",
        k=200,
        is_hra=True,
        channel=None,
    ):
        super().__init__(FL_type, role, channel)
        if self.FL_type == "H" and strategy != "constant":
            self.check_channel()
        self.sketch_name = sketch_name
        self.k = k
        self.is_hra = is_hra
        self.module = SKL_SimpleImputer(
            missing_values=missing_values,
            strategy=strategy,
            fill_value=fill_value,
            copy=copy,
            add_indicator=add_indicator,
            keep_empty_features=keep_empty_features,
        )

    def Hfit(self, X):
        self.module._validate_params()
        validate_quantile_sketch_params(self)

        if self.role == "client":
            X = self.module._validate_input(X, in_fit=True)

            # default fill_value is 0 for numerical input and "missing_value"
            # otherwise
            if self.module.fill_value is None:
                if X.dtype.kind in ("i", "u", "f"):
                    fill_value = 0
                else:
                    fill_value = "missing_value"
            else:
                fill_value = self.module.fill_value

        elif self.role == "server":
            fill_value = self.module.fill_value

        self.module.statistics_ = self._dense_fit(X, self.module.strategy, fill_value)
        return self

    def _dense_fit(self, X, strategy, fill_value):
        """Fit the transformer on dense data."""
        if self.role == "client":
            missing_mask = _get_mask(X, self.module.missing_values)
            masked_X = np.ma.masked_array(X, mask=missing_mask)
            self.module._fit_indicator(missing_mask)

        # Mean
        if strategy == "mean":
            if self.role == "client":
                sum_masked = np.ma.sum(masked_X, axis=0)
                self.channel.send("sum_masked", sum_masked)

                n_samples = X.shape[0] - np.sum(missing_mask, axis=0)
                # for backward-compatibility, reduce n_samples to an integer
                # if the number of samples is the same for each feature (i.e. no
                # missing values)
                if np.ptp(n_samples) == 0:
                    n_samples = n_samples[0]
                self.channel.send("n_samples", n_samples)
                mean = self.channel.recv("mean")

            elif self.role == "server":
                sum_masked = self.channel.recv_all("sum_masked")
                sum_masked = np.ma.sum(sum_masked, axis=0)

                n_samples = self.channel.recv_all("n_samples")
                # n_samples could be np.int or np.ndarray
                n_sum = 0
                for n in n_samples:
                    n_sum += n
                if isinstance(n_sum, np.ndarray) and np.ptp(n_sum) == 0:
                    n_sum = n_sum[0]

                mean_masked = sum_masked / n_sum
                # Avoid the warning "Warning: converting a masked element to nan."
                mean = np.ma.getdata(mean_masked)
                mean[np.ma.getmask(mean_masked)] = (
                    0 if self.module.keep_empty_features else np.nan
                )
                self.channel.send_all("mean", mean)
            return mean

        # Median
        elif strategy == "median":
            if self.role == "client":
                send_local_quantile_sketch(
                    masked_X,
                    self.channel,
                    sketch_name=self.sketch_name,
                    k=self.k,
                    is_hra=self.is_hra,
                )
                median = self.channel.recv("median")

            elif self.role == "server":
                sketch = merge_local_quantile_sketch(
                    channel=self.channel,
                    sketch_name=self.sketch_name,
                    k=self.k,
                    is_hra=self.is_hra,
                )

                if self.sketch_name == "KLL":
                    mask = sketch.is_empty()
                elif self.sketch_name == "REQ":
                    mask = [col_sketch.is_empty() for col_sketch in sketch]

                if not any(mask):
                    median = get_quantiles(
                        quantiles=0.5,
                        sketch=sketch,
                        sketch_name=self.sketch_name,
                    )
                else:
                    median = np.zeros_like(mask, dtype=float)
                    idx = [i for i, x in enumerate(mask) if not x]
                    if self.sketch_name == "KLL":
                        median[idx] = sketch.get_quantiles(0.5, isk=idx).reshape(-1)
                    elif self.sketch_name == "REQ":
                        for i in idx:
                            median[i] = sketch[i].get_quantile(0.5)
                    median[mask] = 0 if self.module.keep_empty_features else np.nan

                self.channel.send_all("median", median)
            return median

        # Most frequent
        elif strategy == "most_frequent":
            if self.role == "client":
                items, counts = [], []
                for col, col_mask in zip(X.T, missing_mask.T):
                    col = col[~col_mask]
                    if len(col) == 0:
                        items.append([])
                        counts.append([])
                    else:
                        col_items, col_counts = _unique(col, return_counts=True)
                        items.append(col_items)
                        counts.append(col_counts)

                send_local_fi_sketch(items, counts, channel=self.channel, k=self.k)
                most_frequent = self.channel.recv("most_frequent")

            elif self.role == "server":
                sketch = merge_local_fi_sketch(
                    channel=self.channel,
                    k=self.k,
                )

                mask = [col_sketch.is_empty() for col_sketch in sketch]
                if not any(mask):
                    most_frequent, _ = get_frequent_items(
                        sketch=sketch,
                        error_type="NFP",
                        max_item=1,
                    )
                    most_frequent = np.array(most_frequent, dtype=object).reshape(-1)
                else:
                    most_frequent = np.empty(len(sketch), dtype=object)
                    for i, empty in enumerate(mask):
                        if empty:
                            most_frequent[i] = (
                                0 if self.module.keep_empty_features else np.nan
                            )
                        else:
                            item, _ = get_frequent_items(
                                sketch[i],
                                error_type="NFP",
                                max_item=1,
                                vector=False,
                            )
                            most_frequent[i] = item[0]
                self.channel.send_all("most_frequent", most_frequent)
            return most_frequent

        # Constant
        elif strategy == "constant":
            if self.role == "client":
                # for constant strategy, self.statistcs_ is used to store
                # fill_value in each column
                return np.full(X.shape[1], fill_value, dtype=X.dtype)
            elif self.role == "server":
                return fill_value

        # Custom
        elif isinstance(strategy, Callable):
            if self.role == "client":
                statistics = strategy(masked_X, self.channel)
            elif self.role == "server":
                statistics = strategy(self.channel)
            return statistics
