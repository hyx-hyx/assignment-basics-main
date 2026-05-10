import torch
from jaxtyping import Float, Int
from tests.conftest import batch_size


def log_softmax(x: torch.Tensor, dim: int):
    assert (dim < len(x.shape))
    x = x - torch.amax(x, dim=dim, keepdim=True)
    log_sum_exp = torch.log(torch.sum(torch.exp(x), dim=dim, keepdim=True))
    return x - log_sum_exp


def eval_cross_entropy(o: Float[torch.Tensor, " batch_size vocab_size"], x: Int[torch.Tensor, " batch_size"]):
    log_softmax_val = -log_softmax(o, dim=1)
    batch_size = x.shape[-1]
    logp = log_softmax_val.gather(dim=1, index=x.view(-1, 1)).squeeze(1)
    return torch.sum(logp) / batch_size
