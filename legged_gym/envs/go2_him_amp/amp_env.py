"""AMP motion capture confined to this subclass, before the original reset."""
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.go2_him.flat_quality_env import Go2HIMFlatQualityRobot
from third_party.himloco_rsl_rl.rsl_rl.amp.features import Go2Kinematics, JOINT_NAMES


class Go2HIMAMPRobot(Go2HIMFlatQualityRobot):
    def __init__(self, *args, **kwargs):
        self.amp_capture_enabled = False
        self._amp_collecting_step = False
        super().__init__(*args, **kwargs)
        if self.cfg.terrain.mesh_type != 'plane':
            raise ValueError('AMP v1 motion features require plane terrain')
        if tuple(self.dof_names) != JOINT_NAMES:
            raise ValueError('AMP environment joint order mismatch')
        self.amp_kinematics = Go2Kinematics(LEGGED_GYM_ROOT_DIR+'/resources/robots/go2/urdf/go2.urdf', self.device)
        self.amp_capture_enabled = True

    def get_amp_observations(self):
        return self.amp_kinematics.features(self.root_states, self.dof_pos, self.dof_vel)

    def reset_idx(self, env_ids):
        if self._amp_collecting_step and self.amp_capture_enabled and len(env_ids):
            self._amp_terminal[env_ids] = self.get_amp_observations()[env_ids]
        return super().reset_idx(env_ids)

    def step(self, actions):
        if not self.amp_capture_enabled:
            return super().step(actions)
        self._amp_terminal = torch.empty(self.num_envs, 43, device=self.device)
        self._amp_collecting_step = True
        try:
            result = super().step(actions)
        finally:
            self._amp_collecting_step = False
        current = self.get_amp_observations()
        terminal_ids = result[5]
        next_motion = current.clone()
        next_motion[terminal_ids] = self._amp_terminal[terminal_ids]
        # Keep the original seven-value step contract, using a private info key.
        result[4]['amp_transition_next'] = next_motion
        return result
