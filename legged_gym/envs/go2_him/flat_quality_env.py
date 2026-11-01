"""Flat-only foot geometry reward; original profiles remain unchanged."""
import torch
from .retrain_env import Go2HIMRetrainRobot


class Go2HIMFlatQualityRobot(Go2HIMRetrainRobot):
    def _reward_joint_pos(self):
        # Match the frozen PPO recipe, including its idle multiplier.
        command_speed = torch.linalg.norm(self.commands[:, :3], dim=1)
        body_speed = torch.linalg.norm(self.base_lin_vel[:, :2], dim=1)
        deviation = torch.linalg.norm(self.dof_pos - self.default_dof_pos, dim=1)
        moving = (command_speed > 0.) | (body_speed > .3)
        return torch.where(moving, deviation, 5. * deviation)

    def _reward_flat_swing_clearance(self):
        if self.cfg.terrain.mesh_type != 'plane':
            raise ValueError('Flat clearance reward requires plane terrain')
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        clearance = self.feet_pos[:, :, 2] - self.cfg.rewards.foot_radius
        target = self.cfg.rewards.swing_clearance_target
        # A bounded, normalized swing-only penalty. This is not a guaranteed
        # swing apex or a hard height constraint; evaluate every foot separately.
        error = torch.clamp((clearance-target)/target, -2., 2.).square()
        moving = torch.linalg.norm(self.commands[:, :3], dim=1) > .05
        return (error * (~contact)).mean(dim=1) * moving
