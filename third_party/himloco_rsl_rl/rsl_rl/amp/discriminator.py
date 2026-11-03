"""Least-squares AMP; formulas follow AMP_for_hardware (see upstream.json)."""
import torch
from torch import nn


class AMPDiscriminator(nn.Module):
    def __init__(self, feature_dim=43, hidden_dims=(1024, 512)):
        super().__init__()
        layers, width = [], feature_dim*2
        for hidden in hidden_dims:
            layers.extend((nn.Linear(width, hidden), nn.ReLU()))
            width = hidden
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(width, 1)

    def forward(self, state, next_state):
        return self.head(self.trunk(torch.cat((state, next_state), -1))).squeeze(-1)

    @torch.no_grad()
    def style_reward(self, state, next_state):
        return (1 - .25*(self(state, next_state)-1).square()).clamp(min=0.)

    def loss(self, policy, expert, gradient_penalty=10.):
        ps, pn = (v.detach() for v in policy)
        es, en = (v.detach().requires_grad_(True) for v in expert)
        pd, ed = self(ps, pn), self(es, en)
        least_squares = .5*((pd+1).square().mean() + (ed-1).square().mean())
        gradients = torch.autograd.grad(ed.sum(), (es, en), create_graph=True)
        penalty = gradient_penalty*torch.cat(gradients, -1).square().sum(-1).mean()
        metrics = dict(discriminator=float(least_squares.detach()), gradient_penalty=float(penalty.detach()),
                       policy_score=float(pd.detach().mean()), expert_score=float(ed.detach().mean()))
        return least_squares+penalty, metrics
