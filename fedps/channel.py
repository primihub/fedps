import zmq
import pickle
from typing import Union


class ClientChannel:
    def __init__(
        self,
        local_ip: str,
        local_port: int,
        remote_ip: str,
        remote_port: int,
    ):
        context = zmq.Context()
        self.addr = f"_{local_ip}:{local_port}"

        self.recv_buffer = {}
        self.recv_channel = context.socket(zmq.PULL)
        self.recv_channel.bind(f"tcp://{local_ip}:{local_port}")

        self.send_channel = context.socket(zmq.PUSH)
        self.send_channel.connect(f"tcp://{remote_ip}:{remote_port}")

    def send(self, key: str, val):
        print(f"send: {key}")
        self.send_channel.send(pickle.dumps((key + self.addr, val)))
        assert self.recv_channel.recv() == b""

    def recv(self, key: str):
        print(f"recv: {key}")
        k, v = pickle.loads(self.recv_channel.recv())
        self.recv_buffer[k] = v
        return self.recv_buffer.pop(key)


class ServerChannel:
    def __init__(
        self,
        local_ip: str,
        local_port: int,
        remote_ip: list,
        remote_port: list,
    ):
        n_ip, n_port = len(remote_ip), len(remote_port)
        if n_ip != n_port:
            raise RuntimeError(
                f"The number of IP and port don't match: "
                f"# ip {n_ip} # port {n_port}"
            )
        self.n_client = n_ip

        context = zmq.Context()

        self.recv_buffer = {}
        self.recv_channel = context.socket(zmq.PULL)
        self.recv_channel.bind(f"tcp://{local_ip}:{local_port}")

        self.client_addr = []
        self.send_channel = []
        for ip, port in zip(remote_ip, remote_port):
            self.client_addr.append(f"_{ip}:{port}")
            socket = context.socket(zmq.PUSH)
            socket.connect(f"tcp://{ip}:{port}")
            self.send_channel.append(socket)

    def send_all(self, key: str, val):
        # Send to all clients with same data
        print(f"send_all: {key}")
        data = pickle.dumps((key, val))
        for channel in self.send_channel:
            channel.send(data)

    def recv_all(self, key: str):
        # Receive from all clients
        print(f"recv_all: {key}")
        for _ in range(self.n_client):
            k, v = pickle.loads(self.recv_channel.recv())
            self.recv_buffer[k] = v

        vals = []
        for addr in self.client_addr:
            vals.append(self.recv_buffer.pop(key + addr))

        for channel in self.send_channel:
            channel.send(b"")
        return vals

    def send_selected(self, key: str, val, idx: Union[int, list[int]]):
        # Send to selected clients with same data via idx
        print(f"send_selected {idx}: {key}")
        data = pickle.dumps((key, val))
        if not hasattr(idx, "__len__"):
            idx = [idx]
        for i in idx:
            self.send_channel[i].send(data)

    def recv_selected(self, key: str, idx: Union[int, list[int]]):
        # Receive from selected clients via idx
        print(f"recv_selected {idx}: {key}")
        is_idx_int = not hasattr(idx, "__len__")
        for _ in range(1 if is_idx_int else len(idx)):
            k, v = pickle.loads(self.recv_channel.recv())
            self.recv_buffer[k] = v

        if is_idx_int:
            vals = self.recv_buffer.pop(key + self.client_addr[idx])
            self.send_channel[idx].send(b"")
        else:
            vals = []
            for i in idx:
                vals.append(self.recv_buffer.pop(key + self.client_addr[i]))
            for i in idx:
                self.send_channel[i].send(b"")
        return vals

    def send_all_diff(self, key: str, val: list):
        # Send to all clients with different data
        print(f"send_all_diff: {key}")
        if len(val) != self.n_client:
            raise ValueError(
                f"The number of elements {len(val)} must equal"
                f" to the number of clients {self.n_client}"
            )

        for channel, v in zip(self.send_channel, val):
            data = pickle.dumps((key, v))
            channel.send(data)

    def send_selected_diff(self, key: str, val: list, idx: list[int]):
        # Send to selected clients with different data via idx
        print(f"send_selected_diff {idx}: {key}")
        if len(val) != len(idx):
            raise ValueError(
                f"The number of elements {len(val)} must equal"
                f" to the number of clients {len(idx)}"
            )

        for i, v in zip(idx, val):
            data = pickle.dumps((key, v))
            self.send_channel[i].send(data)
