## Muon code from Moonlight
## https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py

# This code snippet is a modified version adapted from the following GitHub repository:
# https://github.com/KellerJordan/Muon/blob/master/muon.py
from itertools import repeat
import torch
import math
import warnings
import torch.distributed as dist

polar_express_coeffs_list = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
]

@torch.compile
def PolarExpress(G: torch.Tensor, steps, frob_eps=1e-2, deflation_eps=1e-2):
    assert G.ndim >= 2, "Input tensor must have at least two dimensions."
    X = G 
    if G.size(-2) > G.size(-1):  # opposite convention from our other code
        X = X.mT
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * (1 + frob_eps) + 1e-7)

    hs = polar_express_coeffs_list[:steps] + list(repeat(polar_express_coeffs_list[-1], steps - len(polar_express_coeffs_list)))
    for a, b, c in hs:
        a = a / (1 + deflation_eps)
        b = b / (1 + deflation_eps)
        c = c / (1 + deflation_eps)
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

    def trace_inner_product_eisum(grad: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Computes the trace inner product between `grad` and `v`.
        - If `grad` and `v` are 2D matrices, it computes torch.trace(grad.T @ v).
        - If `grad` and `v` are 3D tensors, it computes the sum of trace inner products
        over the batch dimension.

        Args:
            grad (torch.Tensor): Gradient tensor of shape (dim, dim) or (batch, dim, dim).
            v (torch.Tensor): Tensor of shape (dim, dim) or (batch, dim, dim).

        Returns:
            torch.Tensor: The trace inner product (scalar).
        """
        # Use einsum to compute the trace inner product
        return torch.einsum("...ij,...ji->...", grad, v).sum()

def trace_inner_product(grad: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Computes the trace inner product between `grad` and `v`.
    - If `grad` and `v` are 2D matrices, it computes torch.trace(grad.T @ v).
    - If `grad` and `v` are 3D tensors, it computes the sum of trace inner products
      over the batch dimension.

    Args:
        grad (torch.Tensor): Gradient tensor of shape (dim, dim) or (batch, dim, dim).
        v (torch.Tensor): Tensor of shape (dim, dim) or (batch, dim, dim).

    Returns:
        torch.Tensor: The trace inner product (scalar).
    """
    if grad.ndim == 2 and v.ndim == 2:
        # Case 1: Both are 2D matrices
        return torch.trace(grad.mT @ v)
    elif grad.ndim == 3 and v.ndim == 3:
        # Case 2: Both are 3D tensors
        # Compute the trace for each batch and sum over the batch dimension
        return (grad.mT @ v).diagonal(dim1=-2, dim2=-1).sum(-1).sum()
    else:
        raise ValueError("Both grad and v must have the same dimensions (either 2D or 3D).")

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Warning: This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    """
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        params = list(params)
        sizes = {p.shape for p in params}
        # create one buffer per unique parameter-size
        param_groups = []
        for size in sizes:
            group_params = [p for p in params if p.shape == size]
            param_groups.append(dict(params=group_params))
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        # Efficient systems-wise implementation of step developed by @YouJiacheng,
        # @KonstantinWilleke, @alexrgilbert, @adricarda, @tuttyfrutyee, @vdlad,
        # @ryanyang0, and @vagrawal.
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        reduce_scatter_futures: list[torch.Future] = []
        all_reduce_futures: list[torch.Future] = []
        local_nuc_norm_sum = torch.tensor(0.0, device="cuda")
        # Retrieve the previous global_nuc_norm_sum from the optimizer's state
        if "global_nuc_norm_sum" in self.state:
            prev_global_nuc_norm_sum = self.state["global_nuc_norm_sum"]
        else:
            prev_global_nuc_norm_sum = 1.0
             
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            grad = torch.empty_like(params[-1])
            grad_pad = [param.grad for param in params] + [torch.zeros_like(params[-1])] * world_size
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    grad = params[base_i + rank].grad
                # This gives strange dynamo warnings
                reduce_scatter_futures.append(dist.reduce_scatter(grad, grad_pad[base_i:base_i + world_size], op=dist.ReduceOp.AVG, async_op=True).get_future())

        idx = 0
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * world_size
            momentum = group["momentum"]
            for base_i in range(0, len(params), world_size):
                reduce_scatter_futures[idx].wait()
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    grad = p.grad
                    eff_lr = group["lr"] * max(1, p.size(-2) / p.size(-1)) ** 0.5 * getattr(p, "lr_mul", 1.0)
                    eff_weight_decay = group["lr"] * group["weight_decay"] * getattr(p, "wd_mul", 1.0)
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(grad)
                    momentum_buffer = state["momentum_buffer"]
                    p.mul_(1 - eff_weight_decay)
                    momentum_buffer.lerp_(grad, 1 - momentum)
                    grad = grad.lerp_(momentum_buffer, momentum)
                    # print( " p shape: ", p.shape)
                    v = PolarExpress(grad.bfloat16(), steps=5) #group["ns_steps"]
                    # print("v shape: ", v.shape, "  grad shape: ", grad.shape)
                    # Compute nuc_norm_grad using grad and v
                    nuc_norm_grad = trace_inner_product(grad.bfloat16(), v)
                    # print("nuc_norm_grad: ", nuc_norm_grad)
                    # nuc_norm_grad =  torch.trace(torch.matmul(grad.bfloat16(), v))
                    # nuc_norm_grad = torch.trace(grad.bfloat16().mT @ v) 
                    # Accumulate local sum of nuc_norm_grad
                    local_nuc_norm_sum += nuc_norm_grad
                    eff_lr = prev_global_nuc_norm_sum*eff_lr
                    p.add_(other=v, alpha=-eff_lr)
                idx += 1
                all_reduce_futures.append(dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank], async_op=True).get_future())
        torch.futures.collect_all(all_reduce_futures).wait()
        # Synchronize and sum nuc_norm_grad across all processes
        global_nuc_norm_sum = torch.tensor(0.0, device="cuda")
        dist.all_reduce(local_nuc_norm_sum, op=dist.ReduceOp.SUM, async_op=False)
        global_nuc_norm_sum = local_nuc_norm_sum
        # Store the current global_nuc_norm_sum in the optimizer's state
        self.state["global_nuc_norm_sum"] = global_nuc_norm_sum
        