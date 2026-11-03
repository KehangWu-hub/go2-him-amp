"""Versioned Go2 AMP features, independent of Isaac Gym and actor observations."""
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import torch

FEATURE_VERSION = 'go2_body_motion_43_v1'
FEATURE_DIM = 43
LEGS = ('FL', 'FR', 'RL', 'RR')
JOINT_NAMES = tuple(leg + '_' + joint + '_joint' for leg in LEGS for joint in ('hip', 'thigh', 'calf'))


def quat_rotate_inverse(q, v):
    """Unit quaternion xyzw, world vector to base coordinates."""
    xyz = q[..., :3]
    return v - 2 * q[..., 3:] * torch.cross(xyz, v, dim=-1) + 2 * torch.cross(xyz, torch.cross(xyz, v, dim=-1), dim=-1)


class Go2Kinematics:
    """Foot-link origins from the actual URDF, including freshly reset joints."""
    def __init__(self, urdf, device='cpu'):
        path = Path(urdf)
        self.urdf_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        joints = {j.attrib['name']: j for j in ET.parse(path).getroot().findall('joint')}
        origins, limits = [], []
        for leg in LEGS:
            offsets = []
            for suffix, axis in (('hip', '1 0 0'), ('thigh', '0 1 0'), ('calf', '0 1 0'), ('foot', '0 0 0')):
                joint = joints[leg + '_' + suffix + '_joint']
                origin = joint.find('origin')
                if any(float(x) != 0 for x in origin.attrib.get('rpy', '0 0 0').split()):
                    raise ValueError('Unsupported Go2 joint origin rotation')
                if joint.find('axis').attrib['xyz'] != axis:
                    raise ValueError('Unsupported Go2 joint axis')
                offsets.append([float(x) for x in origin.attrib['xyz'].split()])
                if suffix != 'foot':
                    lim = joint.find('limit')
                    limits.append([float(lim.attrib['lower']), float(lim.attrib['upper'])])
            origins.append(offsets)
        self.origins = torch.tensor(origins, device=device, dtype=torch.float32)
        self.limits = torch.tensor(limits, device=device, dtype=torch.float32)

    def feet(self, q):
        """[batch,12] FL/FR/RL/RR -> [batch,4,3] foot origins in base."""
        q = q.reshape(-1, 4, 3)
        hip, thigh, calf = q.unbind(-1)
        o = self.origins.to(q)
        # R_y(thigh) calf_origin + R_y(thigh+calf) foot_origin.
        def ry(angle, v):
            c, s = torch.cos(angle), torch.sin(angle)
            return torch.stack((c*v[:, 0]+s*v[:, 2], v[:, 1].expand_as(c), -s*v[:, 0]+c*v[:, 2]), -1)
        p = o[:, 1] + ry(thigh, o[:, 2]) + ry(thigh+calf, o[:, 3])
        c, s = torch.cos(hip), torch.sin(hip)
        rotated = torch.stack((p[..., 0], c*p[..., 1]-s*p[..., 2], s*p[..., 1]+c*p[..., 2]), -1)
        return o[:, 0] + rotated

    def features(self, root, q, dq):
        root, q, dq = root.detach(), q.detach(), dq.detach()
        quat = root[:, 3:7]
        norm = torch.linalg.norm(quat, dim=-1, keepdim=True)
        if not bool(torch.isfinite(root).all() and torch.isfinite(q).all() and torch.isfinite(dq).all()) or bool((norm < 1e-6).any()):
            raise ValueError('Invalid AMP physical state')
        quat = quat / norm
        return torch.cat((q, self.feet(q).flatten(1), quat_rotate_inverse(quat, root[:, 7:10]),
                          quat_rotate_inverse(quat, root[:, 10:13]), dq, root[:, 2:3]), -1).clone()
