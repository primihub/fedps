import warnings
import numpy as np
from .min_max import col_max_client, col_max_server, row_max_client, row_max_server
from .sum import col_sum_client, col_sum_server, row_sum_client, row_sum_server
from .util import check_channel, check_FL_type, check_role


def check_norm(norm: str):
    norm = norm.lower()
    valid_norm = {"l1", "l2", "max"}

    if norm in valid_norm:
        return norm
    else:
        raise ValueError(f"Unsupported norm: {norm}, use {valid_norm} instead.")


def col_norm(
    FL_type: str,
    role: str,
    X=None,
    norm: str = "l2",
    ignore_nan: bool = True,
    channel=None,
):
    FL_type = check_FL_type(FL_type)
    role = check_role(role)
    norm = check_norm(norm)

    if FL_type == "H":
        if role == "client":
            return col_norm_client(X, norm, ignore_nan, channel)
        else:
            return col_norm_server(norm, ignore_nan, channel)
    elif role == "client":
        return col_norm_client(
            X, norm, ignore_nan, send_server=False, recv_server=False
        )
    else:
        warnings.warn("Server doesn't have data", RuntimeWarning)


def row_norm(
    FL_type: str,
    role: str,
    X=None,
    norm: str = "l2",
    ignore_nan: bool = True,
    channel=None,
):
    FL_type = check_FL_type(FL_type)
    role = check_role(role)
    norm = check_norm(norm)

    if FL_type == "V":
        if role == "client":
            return row_norm_client(X, norm, ignore_nan, channel)
        else:
            return row_norm_server(norm, ignore_nan, channel)
    elif role == "client":
        return row_norm_client(
            X, norm, ignore_nan, send_server=False, recv_server=False
        )
    else:
        warnings.warn("Server doesn't have data", RuntimeWarning)


def col_norm_client(
    X,
    norm: str = "l2",
    ignore_nan: bool = True,
    channel=None,
    send_server: bool = True,
    recv_server: bool = True,
):
    check_channel(channel, send_server, recv_server)

    if norm == "l1":
        server_col_norm = col_sum_client(
            np.abs(X), ignore_nan, channel, send_server, recv_server
        )
    elif norm == "l2":
        server_col_norm = col_sum_client(
            np.square(X), ignore_nan, channel, send_server, recv_server=False
        )

        if recv_server:
            if not send_server:
                warnings.warn(
                    "server_col_norm=None because send_server=False",
                    RuntimeWarning,
                )
            server_col_norm = channel.recv("server_col_norm")
        else:
            # sqrt it to get the client local l2 norm
            np.sqrt(server_col_norm, server_col_norm)
    else:  # norm == "max"
        server_col_norm = col_max_client(
            np.abs(X), ignore_nan, channel, send_server, recv_server
        )
    return server_col_norm


def col_norm_server(
    norm: str = "l2",
    ignore_nan: bool = True,
    channel=None,
    send_client: bool = True,
    recv_client: bool = True,
):
    check_channel(channel, send_client, recv_client)

    if norm == "l1":
        server_col_norm = col_sum_server(ignore_nan, channel, send_client, recv_client)
    elif norm == "l2":
        server_col_norm = col_sum_server(
            ignore_nan, channel, send_client=False, recv_client=recv_client
        )
        np.sqrt(server_col_norm, server_col_norm)

        if send_client:
            if not recv_client:
                warnings.warn(
                    "server_col_norm=None because recv_client=False",
                    RuntimeWarning,
                )
                channel.send_all("server_col_norm", None)
            else:
                channel.send_all("server_col_norm", server_col_norm)
    else:  # norm == "max"
        server_col_norm = col_max_server(ignore_nan, channel, send_client, recv_client)
    return server_col_norm


def row_norm_client(
    X,
    norm: str = "l2",
    ignore_nan: bool = True,
    channel=None,
    send_server: bool = True,
    recv_server: bool = True,
):
    check_channel(channel, send_server, recv_server)

    if norm == "l1":
        server_row_norm = row_sum_client(
            np.abs(X), ignore_nan, channel, send_server, recv_server
        )
    elif norm == "l2":
        server_row_norm = row_sum_client(
            np.square(X), ignore_nan, channel, send_server, recv_server=False
        )

        if recv_server:
            if not send_server:
                warnings.warn(
                    "server_row_norm=None because send_server=False",
                    RuntimeWarning,
                )
            server_row_norm = channel.recv("server_row_norm")
        else:
            # sqrt it to get the client local l2 norm
            np.sqrt(server_row_norm, server_row_norm)
    else:  # norm == "max"
        server_row_norm = row_max_client(
            np.abs(X), ignore_nan, channel, send_server, recv_server
        )
    return server_row_norm


def row_norm_server(
    norm: str = "l2",
    ignore_nan: bool = True,
    channel=None,
    send_client: bool = True,
    recv_client: bool = True,
):
    check_channel(channel, send_client, recv_client)

    if norm == "l1":
        server_row_norm = row_sum_server(ignore_nan, channel, send_client, recv_client)
    elif norm == "l2":
        server_row_norm = row_sum_server(
            ignore_nan, channel, send_client=False, recv_client=recv_client
        )
        np.sqrt(server_row_norm, server_row_norm)

        if send_client:
            if not recv_client:
                warnings.warn(
                    "server_row_norm=None because recv_client=False",
                    RuntimeWarning,
                )
                channel.send_all("server_row_norm", None)
            else:
                channel.send_all("server_row_norm", server_row_norm)
    else:  # norm == "max"
        server_row_norm = row_max_server(ignore_nan, channel, send_client, recv_client)
    return server_row_norm
