"""
PoE Pipeline Parallelism: split model at PoE detach boundaries across nodes.

Key insight: PoE flat mode detaches gradients at stage boundaries, so the backward
pass is completely local to each pipeline rank. Only forward activations need to
cross the network -- no gradient communication at all.

Communication: TCP sockets for cross-node activation transfer. Within each node,
standard DDP (NCCL) handles multi-GPU data parallelism.

Usage:
    # Node 0 (layers 0..split-1):
    PIPELINE_RANK=0 PIPELINE_PEER=<node1_ip> torchrun --nproc_per_node=4 ...

    # Node 1 (layers split..n_layer-1):
    PIPELINE_RANK=1 PIPELINE_PEER=<node0_ip> torchrun --nproc_per_node=4 ...
"""

import socket
import struct
import torch
import torch.distributed as dist


class PipelineComm:
    """Point-to-point activation transfer over TCP between two pipeline ranks.

    Rank 0 connects to rank 1's server. Each DDP local_rank gets its own
    connection (port + local_rank) so transfers run in parallel.
    """

    def __init__(self, pipeline_rank, peer_addr, base_port=29600, local_rank=0, timeout=300):
        self.rank = pipeline_rank
        self.peer_addr = peer_addr
        self.port = base_port + local_rank
        self.timeout = timeout
        self.sock = None
        self._connect()

    def _connect(self):
        if self.rank == 1:
            # Rank 1 listens (server)
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # 2MB send/recv buffers for large tensor transfers
            server.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2 * 1024 * 1024)
            server.settimeout(self.timeout)
            try:
                server.bind(("0.0.0.0", self.port))
                server.listen(1)
                print(f"[Pipeline rank 1] Listening on port {self.port}...")
                self.sock, addr = server.accept()
                self.sock.settimeout(self.timeout)
                self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                print(f"[Pipeline rank 1] Connected from {addr}")
            finally:
                server.close()
        else:
            # Rank 0 connects (client)
            import time
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2 * 1024 * 1024)
            for attempt in range(self.timeout):
                try:
                    self.sock.connect((self.peer_addr, self.port))
                    print(f"[Pipeline rank 0] Connected to {self.peer_addr}:{self.port}")
                    return
                except ConnectionRefusedError:
                    if attempt % 10 == 0:
                        print(f"[Pipeline rank 0] Waiting for peer on {self.peer_addr}:{self.port}...")
                    time.sleep(1)
            raise RuntimeError(f"Could not connect to pipeline peer {self.peer_addr}:{self.port}")

    def send_tensor(self, tensor):
        """Send a tensor to the peer. Tensor is moved to CPU, serialized as raw bytes."""
        t = tensor.detach().contiguous().to(dtype=torch.bfloat16, device="cpu")
        data = t.numpy().tobytes()
        # Header: ndim (4B) + shape (4B * ndim) + data_size (8B)
        shape = t.shape
        header = struct.pack(f"!I{len(shape)}I Q", len(shape), *shape, len(data))
        self.sock.sendall(header)
        # Send data in chunks to avoid memory pressure
        sent = 0
        while sent < len(data):
            chunk = min(4 * 1024 * 1024, len(data) - sent)  # 4MB chunks
            self.sock.sendall(data[sent:sent + chunk])
            sent += chunk

    def recv_tensor(self, device):
        """Receive a tensor from the peer."""
        # Read header: ndim
        ndim_bytes = self._recv_exact(4)
        ndim = struct.unpack("!I", ndim_bytes)[0]
        # Read shape
        shape_bytes = self._recv_exact(4 * ndim)
        shape = struct.unpack(f"!{ndim}I", shape_bytes)
        # Read data size
        size_bytes = self._recv_exact(8)
        data_size = struct.unpack("!Q", size_bytes)[0]
        # Read data
        data = self._recv_exact(data_size)
        t = torch.frombuffer(bytearray(data), dtype=torch.bfloat16).clone().reshape(shape)
        return t.to(device=device)

    def _recv_exact(self, size):
        """Receive exactly `size` bytes from the socket."""
        buf = bytearray(size)
        view = memoryview(buf)
        received = 0
        while received < size:
            n = self.sock.recv_into(view[received:])
            if n == 0:
                raise ConnectionError("Pipeline peer disconnected")
            received += n
        return bytes(buf)

    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None


def get_pipeline_split(n_layer, poe_every, pipeline_world_size=2):
    """Compute the layer index where the model is split between pipeline ranks.

    Splits at a PoE detach boundary (multiple of poe_every) as close to the
    midpoint as possible. This ensures the backward pass has no cross-rank
    gradient flow.

    Returns: split_layer (int) -- rank 0 gets layers [0, split), rank 1 gets [split, n_layer)
    """
    assert pipeline_world_size == 2, "Only 2-way pipeline supported"
    mid = n_layer // 2
    # Find nearest poe_every boundary to midpoint
    split = round(mid / poe_every) * poe_every
    if split == 0:
        split = poe_every
    if split >= n_layer:
        split = n_layer - poe_every
    return split


def pipeline_forward(model, idx, targets, poe_every, poe_alpha,
                     pipeline_rank, split_layer, comm, device):
    """Forward pass for one pipeline rank.

    Rank 0: embedding + layers [0, split_layer) + send activation
    Rank 1: recv activation + layers [split_layer, n_layer) + final loss

    Each rank computes PoE stage losses locally. No backward communication needed.
    """
    from nanochat.common import COMPUTE_DTYPE
    B, T = idx.size()
    n_layer = model.config.n_layer

    # Rotary embeddings
    cos_sin = model.cos[:, :T], model.sin[:, :T]

    if pipeline_rank == 0:
        # --- Embedding ---
        x = model.transformer.wte(idx)
        x = x.to(COMPUTE_DTYPE)
        x = _norm(x)

        # Smear
        gate = model.smear_lambda.to(x.dtype) * torch.sigmoid(model.smear_gate(x[:, 1:, :24]))
        x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)

        x0 = x

        # --- Layers [0, split_layer) ---
        poe_loss = torch.zeros((), device=device, dtype=torch.float32)
        poe_head_count = 0

        for i in range(split_layer):
            if i > 0 and i % poe_every == 0:
                x = x.detach()
            x = model.resid_lambdas[i] * x + model.x0_lambdas[i] * x0
            ve = model.value_embeds[str(i)](idx).to(x.dtype) if str(i) in model.value_embeds else None
            x = model.transformer.h[i](x, ve, cos_sin, model.window_sizes[i], None)
            if (i + 1) % poe_every == 0:
                poe_loss = poe_loss + torch.utils.checkpoint.checkpoint(
                    model._poe_layer_loss, x, targets,
                    None, None, 0.5, 2.0,
                    use_reentrant=False,
                )
                poe_head_count += 1

        # --- Send activation to rank 1 ---
        comm.send_tensor(x)

        # Touch unused params so DDP doesn't complain about None grads
        unused = 0.0 * model.backout_lambda.sum()
        # Touch rank-1 layer params
        for i in range(split_layer, n_layer):
            for p in model.transformer.h[i].parameters():
                unused = unused + 0.0 * p.sum()
            if str(i) in model.value_embeds:
                unused = unused + 0.0 * model.value_embeds[str(i)].weight.sum()
        # Touch scalars for rank-1 layers
        unused = unused + 0.0 * model.resid_lambdas[split_layer:].sum()
        unused = unused + 0.0 * model.x0_lambdas[split_layer:].sum()

        return (poe_loss + unused) / (poe_head_count ** (1.0 - poe_alpha))

    else:
        # --- Receive activation from rank 0 ---
        x = comm.recv_tensor(device)
        x = x.detach().requires_grad_(True)  # PoE detach at pipeline boundary

        # Reconstruct x0 on rank 1 to match gpt.py (x0 = x is assigned AFTER smear)
        x0 = model.transformer.wte(idx)
        x0 = x0.to(COMPUTE_DTYPE)
        x0 = _norm(x0)
        gate = model.smear_lambda.to(x0.dtype) * torch.sigmoid(model.smear_gate(x0[:, 1:, :24]))
        x0 = torch.cat([x0[:, :1], x0[:, 1:] + gate * x0[:, :-1]], dim=1)

        # --- Layers [split_layer, n_layer) ---
        poe_loss = torch.zeros((), device=device, dtype=torch.float32)
        poe_head_count = 0

        for i in range(split_layer, n_layer):
            if i > split_layer and i % poe_every == 0:
                x = x.detach()
            x = model.resid_lambdas[i] * x + model.x0_lambdas[i] * x0
            ve = model.value_embeds[str(i)](idx).to(x.dtype) if str(i) in model.value_embeds else None
            x = model.transformer.h[i](x, ve, cos_sin, model.window_sizes[i], None)
            if (i + 1) % poe_every == 0 or i == n_layer - 1:
                poe_loss = poe_loss + torch.utils.checkpoint.checkpoint(
                    model._poe_layer_loss, x, targets,
                    None, None, 0.5, 2.0,
                    use_reentrant=False,
                )
                poe_head_count += 1

        # Touch unused rank-0 params
        unused = 0.0 * (model.smear_gate.weight.sum() + model.smear_lambda.sum() + model.backout_lambda.sum())
        for i in range(split_layer):
            for p in model.transformer.h[i].parameters():
                unused = unused + 0.0 * p.sum()
            if str(i) in model.value_embeds:
                unused = unused + 0.0 * model.value_embeds[str(i)].weight.sum()
        unused = unused + 0.0 * model.resid_lambdas[:split_layer].sum()
        unused = unused + 0.0 * model.x0_lambdas[:split_layer].sum()

        return (poe_loss + unused) / (poe_head_count ** (1.0 - poe_alpha))


def _norm(x):
    """RMS norm matching gpt.py's norm()."""
    import torch.nn.functional as F
    return F.rms_norm(x, (x.size(-1),))
