# Go2 AMP motion data

The `go2_him_amp` task reads a reviewed manifest and motion clips from this directory
or an explicit `--amp-manifest` path. Original HIM tasks do not use these files.
No expert dataset is bundled or implicitly downloaded.

Each NPZ clip stores `times` (strictly increasing seconds) and `features` (N × 43):
joint positions (12), base-frame foot-link positions (12), base-frame linear and
angular velocity (3 + 3), joint velocity (12), and base height above a flat z=0
plane (1). Values use SI units and FL/FR/RL/RR joint order. The motion manifest
declares `feature_version: go2_body_motion_43_v1`, `robot: go2`, `coordinates: base`,
`units: SI`, `height_reference: flat_ground_z0`, the actual URDF SHA256,
`joint_names`, `source`, `license`, `reviewed: true`, `kind: expert`, and `clips`.
Each clip entry supplies `file`, `sha256`, and optional positive `weight`.
Separate discontinuous motion segments into separate clips; do not bridge resets.

Use the same Go2 model to review kinematics, velocity consistency and ground
contact before marking data reviewed. Foreign robot motions need retargeting.
Diagnostic clips are accepted only with an explicit diagnostic setting and must
not be described as expert training data. Original motions and generated NPZ
files remain local; retain attribution and check data terms before redistribution.

## Reusing existing Go2 motions

Prefer already prepared Go2 motion files, such as `datasets/mocap_go2/` in
[legged_rl_gym](https://github.com/zihanwang0422/legged_rl_gym). The loader reads
the original AMP 61-column JSON `.txt` files directly. Each manifest clip uses
`format: amp61` and explicitly declares `source_leg_order` as
`[FR, FL, RR, RL]` or `[FL, FR, RL, RR]`. The usual source/checksum/URDF checks
still apply. Files can stay under the ignored `motions/` directory.

This is a field/order adapter: no new retargeting, speed changes, ground
projection or gait generation. Reference coverage and speed ranges still need
to match the intended experiment. Kinematic consistency alone does not
establish dynamic feasibility or policy quality.

## Multidirectional controller recordings

The reviewed local selection can also use the 17 Go2 controller recordings in
[amp_go2](https://github.com/ak1raljl/amp_go2/tree/14cb83f778b367d963d9907bc4334c69e6b85483/datasets/go2_motion).
They cover stance, three forward/backward speeds, two left/right lateral speeds,
and three left/right yaw speeds. The recording script declares FL/FR/RL/RR order,
body-frame velocities, SI units, and a 20 ms frame period.

These recordings use a different Go2 model from this project's URDF. Do not feed
their recorded foot coordinates directly into the discriminator. A reviewed local
NPZ43 selection uses the target URDF to reconstruct foot positions from the
original joint angles, plus one fixed height offset per source clip to account
for leg geometry and foot radius. Joint angles, joint/body velocities and sample
period remain unchanged. Split clips at any frames outside the training soft
joint limits; never interpolate across removed frames. This is an explicit model
feature adaptation, not a byte-identical feature import or a new motion dataset.

Keep original records and checksums under `motions/native17_original/`, derived
features under `motions/native17_features/`, and activate only the reviewed
selection through `manifest.json`. Equal sampling by the seven motion directions
prevents the number of speed variants from determining direction probabilities.
Kinematic replay and short training checks do not establish closed-loop stability
or command coverage. The source does not provide a separate explicit data license;
retain notices and keep the local data out of redistribution.
