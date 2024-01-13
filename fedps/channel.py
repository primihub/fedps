import zmq
import pickle


class ClientChannel:
    def __init__(
        self,
        local_ip: str,
        local_port: int,
        remote_ip: str,
        remote_port: int,
    ):
        context = zmq.Context()

        self.recv_channel = context.socket(zmq.PULL)
        self.recv_channel.bind(f"tcp://{local_ip}:{local_port}")

        self.send_channel = context.socket(zmq.PUSH)
        self.send_channel.connect(f"tcp://{remote_ip}:{remote_port}")

    def send(self, key, val):
        print(f"Start send {key}")
        self.send_channel.send(pickle.dumps(val))
        self.recv_channel.recv()
        print(f"End send {key}")

    def recv(self, key):
        print(f"Start recv {key}")
        val = pickle.loads(self.recv_channel.recv())
        print(f"End recv {key}")
        return val


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

        self.recv_channel = context.socket(zmq.PULL)
        self.recv_channel.bind(f"tcp://{local_ip}:{local_port}")

        self.send_channel = []
        for ip, port in zip(remote_ip, remote_port):
            socket = context.socket(zmq.PUSH)
            socket.connect(f"tcp://{ip}:{port}")
            self.send_channel.append(socket)

    def send_all(self, key, val):
        data = pickle.dumps(val)
        print(f"Start send_all {key}")
        for channel in self.send_channel:
            channel.send(data)
        print(f"End send_all {key}")

    def recv_all(self, key):
        print(f"Start recv_all {key}")
        vals = []
        for _ in range(self.n_client):
            v = pickle.loads(self.recv_channel.recv())
            vals.append(v)

        for channel in self.send_channel:
            channel.send(b"")
        print(f"End recv_all {key}")
        return vals
