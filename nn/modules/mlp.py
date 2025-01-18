import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int=1024, num_hidden: int=2, actv_fn: nn.Module=nn.SiLU(), out_actv_fn: nn.Module=None):
        super().__init__()
        layers = []
        in_features = in_dim
        for _ in range(num_hidden):
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(actv_fn)
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, out_dim))
        if out_actv_fn is not None:
            layers.append(out_actv_fn)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        out = self.net(x.flatten(start_dim=1))
        if self.out_dim == 784:
            out = out.view(out.size(0), 1, 28, 28)
        return out

def MLPEncoder(out_dim: int):
    return MLP(in_dim=784, out_dim=out_dim, num_hidden=3, hidden_dim=512, actv_fn=nn.SiLU(), out_actv_fn=nn.SiLU())

def MLPBottleneck(in_dim: int, out_dim: int, layer_norm: bool=True):
    out_actv_fn = nn.LayerNorm(out_dim) if layer_norm else None
    return MLP(in_dim=in_dim, out_dim=out_dim, num_hidden=2, hidden_dim=512, actv_fn=nn.SiLU(), out_actv_fn=out_actv_fn)

def MLPDecoder(in_dim: int):
    return MLP(in_dim=in_dim, out_dim=784, num_hidden=3, hidden_dim=512, actv_fn=nn.SiLU(), out_actv_fn=nn.Sigmoid())

def MLPActionEncoder():
    return MLP(in_dim=5, out_dim=128, num_hidden=1, hidden_dim=128, actv_fn=nn.SiLU(), out_actv_fn=None)