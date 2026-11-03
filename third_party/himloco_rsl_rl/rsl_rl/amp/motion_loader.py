"""Validated Go2 feature clips; never silently treats foreign motions as Go2."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .features import FEATURE_DIM, FEATURE_VERSION, JOINT_NAMES


class MotionDataset:
    def __init__(self, manifest, kinematics, step_dt=.02, device='cpu', allow_diagnostic=False):
        path = Path(manifest).resolve()
        data = json.loads(path.read_text())
        if data.get('feature_version') != FEATURE_VERSION or data.get('robot') != 'go2':
            raise ValueError('AMP data robot/feature version mismatch')
        if data.get('urdf_sha256') != kinematics.urdf_sha256 or data.get('joint_names') != list(JOINT_NAMES):
            raise ValueError('AMP data URDF/joint order mismatch')
        if data.get('coordinates') != 'base' or data.get('units') != 'SI' or data.get('height_reference') != 'flat_ground_z0':
            raise ValueError('AMP data coordinate contract mismatch')
        kind = data.get('kind')
        if kind != 'expert' and not (allow_diagnostic and kind == 'diagnostic'):
            raise ValueError('Diagnostic motions are not expert training data')
        if not data.get('source') or not data.get('license') or data.get('reviewed') is not True:
            raise ValueError('AMP data requires source, license and completed review')
        if not data.get('clips'):
            raise ValueError('AMP manifest has no motion clips')
        self.clips, self.ranges, weights = [], [], []
        self.kinematics = kinematics
        self.step_dt, self.device = float(step_dt), device
        if not np.isfinite(self.step_dt) or self.step_dt <= 0:
            raise ValueError('Invalid AMP transition period')
        identities = []
        for clip in data['clips']:
            file = (path.parent/clip['file']).resolve()
            digest = hashlib.sha256(file.read_bytes()).hexdigest()
            if digest != clip['sha256']:
                raise ValueError('AMP motion checksum mismatch: '+str(file))
            if clip.get('format', 'npz43') == 'amp61':
                # Read the existing AMP_for_hardware format directly. Preserve
                # source trajectories; only arrange fields and joint order.
                motion = json.loads(file.read_text())
                frames = np.asarray(motion['Frames'], dtype=np.float32)
                dt = float(motion['FrameDuration'])
                if frames.ndim != 2 or frames.shape[1] != 61 or not np.isfinite(frames).all() or not np.isfinite(dt) or dt <= 0:
                    raise ValueError('Invalid AMP61 reference clip')
                order = clip.get('source_leg_order')
                if order == ['FR', 'FL', 'RR', 'RL']:
                    indices = [1, 0, 3, 2]
                elif order == ['FL', 'FR', 'RL', 'RR']:
                    indices = [0, 1, 2, 3]
                else:
                    raise ValueError('AMP61 source leg order must be explicit')
                def legs(start):
                    return frames[:, start:start+12].reshape(-1, 4, 3)[:, indices].reshape(-1, 12)
                features = np.concatenate((legs(7), legs(19), frames[:, 31:37], legs(37), frames[:, 2:3]), -1)
                times = np.arange(len(frames), dtype=np.float64)*dt
            elif clip.get('format', 'npz43') == 'npz43':
                with np.load(file, allow_pickle=False) as archive:
                    features = np.asarray(archive['features'], dtype=np.float32)
                    times = np.asarray(archive['times'], dtype=np.float64)
            else:
                raise ValueError('Unsupported AMP motion format')
            if features.ndim != 2 or features.shape[1] != FEATURE_DIM or len(features) < 2 or times.shape != (len(features),):
                raise ValueError('Invalid AMP clip dimensions')
            if not np.isfinite(features).all() or not np.isfinite(times).all() or not (np.diff(times) > 0).all():
                raise ValueError('Non-finite or non-monotonic AMP clip')
            if np.diff(times).max() > .1:
                raise ValueError('AMP clip contains a temporal gap; split discontinuous segments')
            if times[-1]-times[0] < self.step_dt:
                raise ValueError('AMP clip shorter than one transition')
            q = torch.as_tensor(features[:, :12], device=kinematics.limits.device)
            if bool(((q < kinematics.limits[:, 0]-.02) | (q > kinematics.limits[:, 1]+.02)).any()):
                raise ValueError('AMP reference joint limits exceeded')
            expected = kinematics.feet(q).flatten(1).cpu().numpy()
            if np.max(np.abs(expected-features[:, 12:24])) > .005:
                raise ValueError('AMP reference foot FK mismatch (>5mm)')
            if (features[:, -1] <= 0).any():
                raise ValueError('AMP reference base below plane')
            weight = float(clip.get('weight', 1.))
            if not np.isfinite(weight) or weight <= 0:
                raise ValueError('Invalid AMP clip sampling weight')
            self.clips.append((torch.tensor(features, device=device), torch.tensor(times, dtype=torch.float64, device=device)))
            self.ranges.append((float(times[0]), float(times[-1])-self.step_dt))
            weights.append(weight)
            identities.append((clip['file'], digest))
        self.weights = torch.tensor(weights, device=device)/sum(weights)
        self.identity = hashlib.sha256(json.dumps(dict(manifest=data, clips=identities), sort_keys=True).encode()).hexdigest()
        self.manifest, self.metadata = str(path), data

    def sample(self, count, generator=None):
        ids = torch.multinomial(self.weights, count, replacement=True, generator=generator)
        states = torch.empty(count, FEATURE_DIM, device=self.device)
        next_states = torch.empty_like(states)
        for i, ((frames, times), (start, end)) in enumerate(zip(self.clips, self.ranges)):
            mask = ids == i
            n = int(mask.sum())
            if n == 0:
                continue
            t = start+torch.rand(n, dtype=torch.float64, device=self.device, generator=generator)*(end-start)
            for query, out in ((t, states), (t+self.step_dt, next_states)):
                hi = torch.searchsorted(times, query).clamp(1, len(times)-1)
                lo = hi-1
                alpha = ((query-times[lo])/(times[hi]-times[lo])).float().unsqueeze(-1)
                out[mask] = frames[lo]+alpha*(frames[hi]-frames[lo])
        # Interpolate joints in time, then reconstruct nonlinear FK identically
        # to the policy side instead of linearly interpolating foot positions.
        for out in (states, next_states):
            out[:, 12:24] = self.kinematics.feet(out[:, :12]).flatten(1)
        return states, next_states
