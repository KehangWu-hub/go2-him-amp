"""AMP adapters around the unchanged HIM runner and HIMPPO implementation."""
import copy
import math
from pathlib import Path
import random

import numpy as np
import torch

from ..runners.him_on_policy_runner import HIMOnPolicyRunner
from .discriminator import AMPDiscriminator
from .features import FEATURE_VERSION
from .motion_loader import MotionDataset
from .normalizer import MotionNormalizer
from .replay_buffer import MotionReplay


class AMPTrainingState:
    def __init__(self, env, config, device):
        self.config, self.device = copy.deepcopy(config), device
        self.weight, self.dt = float(config['weight']), float(env.dt)
        if not math.isfinite(self.weight) or self.weight <= 0 or not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError('Invalid AMP reward weight/period')
        self.data = MotionDataset(config['manifest'], env.amp_kinematics, self.dt, device,
                                  allow_diagnostic=config.get('allow_diagnostic_data', False))
        self.generator = torch.Generator(device=device).manual_seed(int(config['seed']))
        # AMP initialization must not change the existing HIM/environment RNG.
        devices = [torch.device(device).index or 0] if torch.device(device).type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(config['seed']))
            self.discriminator = AMPDiscriminator(hidden_dims=config['hidden_dims']).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.discriminator.trunk.parameters(), 'weight_decay': 1e-3},
            {'params': self.discriminator.head.parameters(), 'weight_decay': .1},
        ], lr=float(config['learning_rate']))
        self.normalizer = MotionNormalizer(43, device)
        raw = torch.cat([frames for frames, times in self.data.clips])
        self.normalizer.update(raw)
        self.replay = MotionReplay(int(config['replay_capacity']), 43, device)
        if int(config['updates_per_iteration']) <= 0 or int(config['batch_size']) <= 0:
            raise ValueError('AMP update counts must be positive')
        self.metrics, self.reward_sums, self.reward_count = {}, torch.zeros(3, device=device), 0

    @torch.no_grad()
    def reward(self, previous, next_state, task):
        previous, next_state = previous.to(self.device), next_state.to(self.device)
        style = self.discriminator.style_reward(self.normalizer(previous), self.normalizer(next_state))
        contribution = self.weight*self.dt*style
        self.replay.insert(previous, next_state)
        self.reward_sums += torch.stack((task.mean(), style.mean(), contribution.mean()))
        self.reward_count += 1
        return task + contribution

    def update(self):
        totals = {}
        for _ in range(int(self.config['updates_per_iteration'])):
            policy = self.replay.sample(int(self.config['batch_size']), self.generator)
            expert = self.data.sample(int(self.config['batch_size']), self.generator)
            # Update with RAW features, never statistics of normalized data.
            self.normalizer.update(torch.cat((*policy, *expert)))
            p = tuple(self.normalizer(v).detach() for v in policy)
            e = tuple(self.normalizer(v).detach() for v in expert)
            loss, metrics = self.discriminator.loss(p, e, float(self.config['gradient_penalty']))
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('Non-finite AMP loss')
            self.optimizer.zero_grad()
            loss.backward()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in self.discriminator.parameters()):
                raise RuntimeError('Non-finite AMP discriminator gradient')
            self.optimizer.step()
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.)+value/int(self.config['updates_per_iteration'])
        rewards = (self.reward_sums/max(1, self.reward_count)).tolist()
        totals.update(zip(('task_reward', 'style_reward', 'style_contribution'), rewards))
        self.metrics = totals
        self.reward_sums.zero_(); self.reward_count = 0

    def state_dict(self):
        return dict(feature_version=FEATURE_VERSION, data_identity=self.data.identity, config=self.config,
                    step_dt=self.dt, discriminator=self.discriminator.state_dict(), optimizer=self.optimizer.state_dict(),
                    normalizer=self.normalizer.state_dict(), replay=self.replay.state_dict(),
                    generator=self.generator.get_state())

    def validate_checkpoint(self, state):
        if state.get('feature_version') != FEATURE_VERSION or state.get('data_identity') != self.data.identity:
            raise ValueError('AMP checkpoint data/feature identity mismatch')
        old, new = dict(state['config']), dict(self.config)
        old.pop('manifest', None); new.pop('manifest', None)
        if old != new or state['step_dt'] != self.dt:
            raise ValueError('AMP checkpoint training configuration mismatch')

    def load_state_dict(self, state):
        self.validate_checkpoint(state)
        self.discriminator.load_state_dict(state['discriminator'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.normalizer.load_state_dict(state['normalizer'])
        self.replay.load_state_dict(state['replay'])
        self.generator.set_state(state['generator'].cpu())


class _AMPEnvAdapter:
    def __init__(self, env, amp):
        self.base, self.amp = env, amp

    def __getattr__(self, name):
        return getattr(self.base, name)

    def step(self, actions):
        previous = self.base.get_amp_observations().clone()
        result = list(self.base.step(actions))
        next_motion = result[4].pop('amp_transition_next')
        result[2] = self.amp.reward(previous, next_motion, result[2])
        return tuple(result)

    # Original runner can randomize the environment episode-length buffer.
    @property
    def episode_length_buf(self):
        return self.base.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.base.episode_length_buf = value


class _AMPAlgorithmAdapter:
    def __init__(self, algorithm, amp):
        self.base, self.amp = algorithm, amp

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __setattr__(self, name, value):
        if name in ('base', 'amp'):
            object.__setattr__(self, name, value)
        else:
            setattr(self.base, name, value)

    def update(self):
        losses = self.base.update()
        self.amp.update()
        return losses


class HIMAMPOnPolicyRunner(HIMOnPolicyRunner):
    def __new__(cls, env, train_cfg, log_dir=None, device='cpu'):
        weight = float(train_cfg['amp']['weight'])
        if not math.isfinite(weight) or weight < 0:
            raise ValueError('AMP weight must be finite and nonnegative')
        if weight == 0:
            env.amp_capture_enabled = False
            return HIMOnPolicyRunner(env, train_cfg, log_dir, device)
        return super().__new__(cls)

    def __init__(self, env, train_cfg, log_dir=None, device='cpu'):
        super().__init__(env, train_cfg, log_dir, device)
        self.amp = AMPTrainingState(env, train_cfg['amp'], device)
        self.env = _AMPEnvAdapter(env, self.amp)
        self.alg = _AMPAlgorithmAdapter(self.alg, self.amp)

    def log(self, locs, *args, **kwargs):
        super().log(locs, *args, **kwargs)
        for name, value in self.amp.metrics.items():
            if not math.isfinite(value):
                raise RuntimeError('Non-finite AMP metric '+name)
            self.writer.add_scalar('AMP/'+name, value, locs['it'])

    def save(self, path, infos=None):
        # Base runner's checkpoint format is preserved; AMP is one versioned key.
        base_temporary = str(path)+'.him.tmp'
        super().save(base_temporary, infos)
        checkpoint = torch.load(base_temporary, map_location=self.device)
        checkpoint['amp'] = self.amp.state_dict()
        checkpoint['rng'] = dict(torch=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate(),
                                 cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
        temporary = str(path)+'.tmp'
        torch.save(checkpoint, temporary)
        Path(temporary).replace(path)
        Path(base_temporary).unlink()

    def load(self, path, load_optimizer=True):
        checkpoint = torch.load(path, map_location=self.device)
        if load_optimizer:
            if 'amp' not in checkpoint or 'rng' not in checkpoint:
                raise ValueError('Full AMP resume requires discriminator, statistics, replay and RNG state; use weights-only explicitly')
            self.amp.validate_checkpoint(checkpoint['amp'])
        infos = super().load(path, load_optimizer)
        if load_optimizer:
            self.amp.load_state_dict(checkpoint['amp'])
            rng = checkpoint['rng']
            torch.set_rng_state(rng['torch'].cpu()); np.random.set_state(rng['numpy']); random.setstate(rng['python'])
            if rng['cuda']:
                torch.cuda.set_rng_state_all([v.cpu() for v in rng['cuda']])
        else:
            # Reset all AMP state even when weights-only is requested on a used runner.
            self.alg.optimizer.state.clear()
            self.alg.actor_critic.estimator.optimizer.state.clear()
            self.amp = AMPTrainingState(self.env.base, self.amp.config, self.device)
            self.env.amp = self.amp
            self.alg.amp = self.amp
            self.current_learning_iteration = 0
        return infos
