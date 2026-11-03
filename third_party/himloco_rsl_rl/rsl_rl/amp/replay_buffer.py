"""Bounded transition replay exclusively for the discriminator, never PPO."""
import torch


class MotionReplay:
    def __init__(self, capacity, feature_dim, device='cpu'):
        if capacity <= 0:
            raise ValueError('AMP replay capacity must be positive')
        self.states = torch.zeros(capacity, feature_dim, device=device)
        self.next_states = torch.zeros_like(self.states)
        self.capacity, self.size, self.cursor = capacity, 0, 0

    @torch.no_grad()
    def insert(self, state, next_state):
        if state.ndim != 2 or state.shape != next_state.shape or state.shape[1] != self.states.shape[1]:
            raise ValueError('Invalid AMP replay shape')
        if not bool(torch.isfinite(state).all() and torch.isfinite(next_state).all()):
            raise ValueError('Non-finite AMP transition')
        state, next_state = state[-self.capacity:], next_state[-self.capacity:]
        indices = (torch.arange(len(state), device=self.states.device)+self.cursor) % self.capacity
        self.states[indices] = state.detach()
        self.next_states[indices] = next_state.detach()
        self.cursor = (self.cursor+len(state)) % self.capacity
        self.size = min(self.capacity, self.size+len(state))

    def sample(self, count, generator=None):
        if not self.size:
            raise ValueError('Empty AMP replay')
        i = torch.randint(self.size, (count,), device=self.states.device, generator=generator)
        return self.states[i], self.next_states[i]

    def state_dict(self):
        return dict(states=self.states[:self.size].clone(), next_states=self.next_states[:self.size].clone(),
                    capacity=self.capacity, cursor=self.cursor, size=self.size)

    def load_state_dict(self, state):
        if state['capacity'] != self.capacity or not 0 <= state['size'] <= self.capacity or not 0 <= state['cursor'] < self.capacity:
            raise ValueError('AMP replay contract mismatch')
        n = state['size']
        if state['states'].shape != (n, self.states.shape[1]) or state['next_states'].shape != state['states'].shape:
            raise ValueError('AMP replay feature shape mismatch')
        self.states[:n].copy_(state['states']); self.next_states[:n].copy_(state['next_states'])
        self.size, self.cursor = n, state['cursor']
