"""One shared running statistic for raw expert and policy motion features."""
import torch


class MotionNormalizer(torch.nn.Module):
    def __init__(self, size, device='cpu'):
        super().__init__()
        self.register_buffer('mean', torch.zeros(size, device=device))
        self.register_buffer('var', torch.ones(size, device=device))
        self.register_buffer('count', torch.tensor(0., dtype=torch.float64, device=device))

    @torch.no_grad()
    def update(self, raw):
        raw = raw.detach()
        if raw.ndim != 2 or raw.shape[1:] != self.mean.shape or not bool(torch.isfinite(raw).all()) or len(raw) == 0:
            raise ValueError('Invalid raw AMP normalization batch')
        mean, var = raw.mean(0), raw.var(0, unbiased=False)
        n = len(raw)
        total = self.count + n
        delta = mean-self.mean
        self.var.copy_((self.var*self.count + var*n + delta.square()*self.count*n/total)/total)
        self.mean.add_(delta*(n/total))
        self.count.copy_(total)

    def forward(self, raw):
        return ((raw-self.mean)/torch.sqrt(self.var+1e-4)).clamp(-10, 10)
