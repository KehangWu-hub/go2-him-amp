"""AMP data, gradient, reset and resume contracts without robot or Isaac Gym."""
import ast
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace as NS

import mujoco
import numpy as np
import torch

from third_party.himloco_rsl_rl.rsl_rl.amp.features import Go2Kinematics, FEATURE_VERSION, JOINT_NAMES
from third_party.himloco_rsl_rl.rsl_rl.amp.discriminator import AMPDiscriminator
from third_party.himloco_rsl_rl.rsl_rl.amp.motion_loader import MotionDataset
from third_party.himloco_rsl_rl.rsl_rl.amp.normalizer import MotionNormalizer
from third_party.himloco_rsl_rl.rsl_rl.amp.replay_buffer import MotionReplay
from third_party.himloco_rsl_rl.rsl_rl.amp.runner import HIMAMPOnPolicyRunner
from third_party.himloco_rsl_rl.rsl_rl.runners import HIMOnPolicyRunner

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT/'resources/robots/go2/urdf/go2.urdf'
DEFAULT = torch.tensor([.1, .8, -1.5, -.1, .8, -1.5, .1, 1., -1.5, -.1, 1., -1.5])


def diagnostic_dataset(directory, device='cpu'):
    """A synthetic contract fixture, explicitly not a reference gait."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    kin = Go2Kinematics(URDF, device)
    times = torch.arange(40, device=device)*.02
    q = DEFAULT.to(device).repeat(40, 1)+.02*torch.sin(times[:, None]*3)
    dq = .06*torch.cos(times[:, None]*3).expand_as(q)
    root = torch.zeros(40, 13, device=device); root[:, 2] = .32; root[:, 6] = 1
    root[:, 7] = .2
    features = kin.features(root, q, dq).cpu().numpy()
    np.savez(path/'diagnostic.npz', features=features, times=times.cpu().numpy())
    manifest = dict(feature_version=FEATURE_VERSION, robot='go2', urdf_sha256=kin.urdf_sha256,
                    joint_names=list(JOINT_NAMES), coordinates='base', units='SI', height_reference='flat_ground_z0',
                    source='synthetic unit-test fixture', license='project test fixture', reviewed=True, kind='diagnostic',
                    clips=[dict(file='diagnostic.npz', sha256=hashlib.sha256((path/'diagnostic.npz').read_bytes()).hexdigest())])
    (path/'manifest.json').write_text(json.dumps(manifest))
    return path/'manifest.json'


def runner_config(manifest, weight=1.):
    return dict(runner=dict(policy_class_name='BoundedHIMActorCritic', algorithm_class_name='HIMPPO',
                            num_steps_per_env=4, save_interval=50),
                policy=dict(actor_hidden_dims=[16, 8], critic_hidden_dims=[16, 8], init_noise_std=.6, action_bound=4.),
                algorithm=dict(num_learning_epochs=1, num_mini_batches=2, learning_rate=.0003, schedule='fixed'),
                amp=dict(weight=weight, manifest=str(manifest), hidden_dims=[16, 8], learning_rate=.0003,
                         gradient_penalty=10., replay_capacity=64, batch_size=8, updates_per_iteration=2,
                         seed=123, allow_diagnostic_data=True))


class FakeMotionEnv:
    num_envs, num_obs, num_privileged_obs, num_one_step_obs, num_actions = 4, 270, 238, 45, 12
    dt, device, max_episode_length = .02, 'cpu', 100
    def __init__(self):
        self.amp_kinematics = Go2Kinematics(URDF)
        self.episode_length_buf = torch.zeros(4, dtype=torch.long)
        self.q, self.dq = DEFAULT.repeat(4, 1), torch.zeros(4, 12)
        self.root = torch.zeros(4, 13); self.root[:, 2] = .32; self.root[:, 6] = 1
        self.obs, self.priv = torch.zeros(4, 270), torch.zeros(4, 238)
        self.ticks, self.amp_capture_enabled = 0, True

    def reset(self):
        return self.obs, self.priv

    def get_observations(self):
        return self.obs

    def get_privileged_observations(self):
        return self.priv

    def get_amp_observations(self):
        return self.amp_kinematics.features(self.root, self.q, self.dq)

    def step(self, actions):
        self.ticks += 1
        self.dq = actions*.01
        self.q = self.q+self.dq*self.dt
        self.obs = torch.cat((actions.repeat(1, 3), torch.zeros(4, 9), self.obs[:, :225]), -1)
        self.priv = torch.cat((self.obs[:, :45], torch.zeros(4, 193)), -1)
        done = torch.zeros(4, dtype=torch.long)
        infos = {'amp_transition_next': self.get_amp_observations()}
        return self.obs, self.priv, torch.ones(4)*.02, done, infos, torch.empty(0, dtype=torch.long), self.priv[:0]


class AMPContractTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.manifest = diagnostic_dataset(self.temp.name)
        self.kin = Go2Kinematics(URDF)

    def test_fk_matches_mujoco_foot_origins_in_random_poses(self):
        model = mujoco.MjModel.from_xml_path(str(ROOT/'resources/robots/go2/mjcf/scene.xml'))
        data = mujoco.MjData(model)
        generator = torch.Generator().manual_seed(30)
        for _ in range(10):
            q = DEFAULT+torch.randn(12, generator=generator)*.15
            data.qpos[:3] = [.3, -.2, .5]; data.qpos[3:7] = [1, 0, 0, 0]
            for name, value in zip(JOINT_NAMES, q):
                data.qpos[model.joint(name).qposadr[0]] = float(value)
            mujoco.mj_forward(model, data)
            feet = np.stack([data.body(leg+'_foot').xpos-data.qpos[:3] for leg in ('FL', 'FR', 'RL', 'RR')])
            np.testing.assert_allclose(self.kin.feet(q[None])[0].numpy(), feet, atol=1e-6)

    def test_amp_configuration_preserves_approved_him_recipe(self):
        from tests.test_go2_him_candidate import candidate_scope
        scope = candidate_scope()
        for relative in ('go2_him/flat_bounded_config.py', 'go2_him_amp/amp_config.py'):
            path = ROOT/'legged_gym/envs'/relative
            tree = ast.parse(path.read_text()); tree.body = [n for n in tree.body if isinstance(n, ast.ClassDef)]
            exec(compile(tree, str(path), 'exec'), scope)
        base, amp = scope['Go2HIMFlatBoundedCfg'], scope['Go2HIMAMPCfg']
        for field in ('env', 'control', 'sim', 'commands', 'rewards', 'domain_rand', 'normalization'):
            self.assertIs(getattr(amp, field), getattr(base, field))
        base_train, amp_train = scope['Go2HIMFlatBoundedCfgPPO'], scope['Go2HIMAMPCfgPPO']
        self.assertIs(amp_train.policy, base_train.policy)
        self.assertIs(amp_train.algorithm, base_train.algorithm)
        self.assertEqual(amp_train.runner_class_name, 'HIMAMPOnPolicyRunner')
        self.assertEqual(base_train.runner_class_name, 'HIMOnPolicyRunner')
        self.assertEqual(amp_train.runner.num_steps_per_env, base_train.runner.num_steps_per_env)

    def test_feature_velocity_rotates_world_to_body(self):
        root = torch.zeros(1, 13); root[0, 2] = .32
        root[0, 5:7] = 2**-.5; root[0, 7] = 1.
        features = self.kin.features(root, DEFAULT[None], torch.zeros(1, 12))
        torch.testing.assert_close(features[0, 24:27], torch.tensor([0., -1., 0.]), atol=1e-6, rtol=0)

    def test_dataset_rejects_diagnostic_by_default_and_wrong_identity(self):
        with self.assertRaisesRegex(ValueError, 'not expert'):
            MotionDataset(self.manifest, self.kin)
        original = json.loads(self.manifest.read_text())
        for key, bad in [('urdf_sha256', 'bad'), ('joint_names', list(reversed(JOINT_NAMES))),
                         ('reviewed', False), ('coordinates', 'world')]:
            with self.subTest(key=key):
                data = dict(original); data[key] = bad; self.manifest.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    MotionDataset(self.manifest, self.kin, allow_diagnostic=True)
        self.manifest.write_text(json.dumps(original))
        file = self.manifest.parent/'diagnostic.npz'
        file.write_bytes(file.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            MotionDataset(self.manifest, self.kin, allow_diagnostic=True)

    def test_samples_use_real_time_and_do_not_cross_clip_boundaries(self):
        dataset = MotionDataset(self.manifest, self.kin, allow_diagnostic=True)
        generator = torch.Generator().manual_seed(1)
        s, n = dataset.sample(128, generator)
        self.assertEqual(s.shape, (128, 43)); self.assertTrue(bool(torch.isfinite(n).all()))
        self.assertTrue(bool(((n[:, 0]-s[:, 0]).abs() <= .06*.02+1e-5).all()))
        self.assertTrue(bool((s[:, -1] == .32).all()))
        torch.testing.assert_close(s[:, 12:24], self.kin.feet(s[:, :12]).flatten(1), rtol=0, atol=0)

    def test_normalizer_uses_raw_population_variance_and_roundtrips(self):
        normalizer = MotionNormalizer(2)
        normalizer.update(torch.tensor([[1., 10.]])); normalizer.update(torch.tensor([[3., 14.]]))
        torch.testing.assert_close(normalizer.mean, torch.tensor([2., 12.]))
        torch.testing.assert_close(normalizer.var, torch.tensor([1., 4.]))
        other = MotionNormalizer(2); other.load_state_dict(normalizer.state_dict())
        torch.testing.assert_close(other(torch.tensor([[2., 12.]])), torch.zeros(1, 2))

    def test_replay_wraps_and_restores_same_samples_without_aliasing(self):
        replay = MotionReplay(5, 2)
        x = torch.arange(16).reshape(8, 2).float()
        replay.insert(x[:3], x[:3]+1); replay.insert(x[3:], x[3:]+1)
        before = replay.states.clone(); x.zero_()
        torch.testing.assert_close(replay.states, before)
        restored = MotionReplay(5, 2); restored.load_state_dict(replay.state_dict())
        a = replay.sample(10, torch.Generator().manual_seed(2))
        b = restored.sample(10, torch.Generator().manual_seed(2))
        for x, y in zip(a, b): torch.testing.assert_close(x, y, rtol=0, atol=0)

    def test_style_formula_penalty_and_gradient_isolation(self):
        disc = AMPDiscriminator(2, [])
        with torch.no_grad(): disc.head.weight.fill_(.5); disc.head.bias.zero_()
        x = torch.zeros(3, 2, requires_grad=True)
        torch.testing.assert_close(disc.style_reward(x, x), torch.ones(3)*.75)
        loss, metrics = disc.loss((x, x), (x, x), 10)
        self.assertAlmostEqual(metrics['gradient_penalty'], 10.)
        loss.backward()
        self.assertIsNone(x.grad); self.assertIsNotNone(disc.head.weight.grad)

    def test_zero_weight_is_exact_original_him_and_does_not_load_data(self):
        config = runner_config('/nonexistent/expert.json', 0)
        with contextlib.redirect_stdout(io.StringIO()):
            torch.manual_seed(19); base = HIMOnPolicyRunner(FakeMotionEnv(), config)
            base.learn(1); expected_rng = torch.get_rng_state().clone()
            torch.manual_seed(19); disabled = HIMAMPOnPolicyRunner(FakeMotionEnv(), config)
            self.assertIs(type(disabled), HIMOnPolicyRunner)
            self.assertFalse(disabled.env.amp_capture_enabled)
            disabled.learn(1)
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
        for name, tensor in base.alg.actor_critic.state_dict().items():
            torch.testing.assert_close(tensor, disabled.alg.actor_critic.state_dict()[name], rtol=0, atol=0)

    def test_full_training_resume_export_and_weights_only(self):
        config = runner_config(self.manifest)
        with contextlib.redirect_stdout(io.StringIO()):
            torch.manual_seed(10); runner = HIMAMPOnPolicyRunner(FakeMotionEnv(), config)
            actor = copy.deepcopy(runner.alg.actor_critic.actor.state_dict())
            estimator = copy.deepcopy(runner.alg.actor_critic.estimator.state_dict())
            discriminator = copy.deepcopy(runner.amp.discriminator.state_dict())
            runner.learn(1)
            for before, after in [(actor, runner.alg.actor_critic.actor.state_dict()),
                                  (estimator, runner.alg.actor_critic.estimator.state_dict()),
                                  (discriminator, runner.amp.discriminator.state_dict())]:
                self.assertTrue(any(not torch.equal(before[k], after[k]) for k in before))
            self.assertEqual(runner.amp.replay.size, 16)
            self.assertAlmostEqual(runner.amp.metrics['style_contribution'], runner.amp.metrics['style_reward']*.02, places=7)
            checkpoint = Path(self.temp.name)/'model.pt'; runner.save(checkpoint)
            expected = runner.amp.data.sample(8, runner.amp.generator)
            restored = HIMAMPOnPolicyRunner(FakeMotionEnv(), config); restored.load(checkpoint)
            actual = restored.amp.data.sample(8, restored.amp.generator)
            for a, b in zip(expected, actual): torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertEqual(restored.current_learning_iteration, 1)
            self.assertTrue(restored.amp.optimizer.state)
            self.assertTrue(restored.alg.optimizer.state)
            self.assertTrue(restored.alg.actor_critic.estimator.optimizer.state)
            from tests.test_go2_him_training import EXPORT
            exported = EXPORT.export_him_policy(restored.alg.actor_critic, Path(self.temp.name)/'export')
            model = torch.jit.load(exported); inputs = torch.randn(8, 270)*.1
            with torch.no_grad():
                torch.testing.assert_close(model(inputs), restored.alg.actor_critic.act_inference(inputs), rtol=0, atol=0)
            state = torch.load(checkpoint); state.pop('amp'); torch.save(state, checkpoint)
            with self.assertRaisesRegex(ValueError, 'Full AMP resume'):
                restored.load(checkpoint)
            restored.load(checkpoint, load_optimizer=False)
            self.assertEqual(restored.current_learning_iteration, 0)
            self.assertFalse(restored.alg.optimizer.state)
            self.assertFalse(restored.alg.actor_critic.estimator.optimizer.state)
            self.assertFalse(restored.amp.optimizer.state)

    def test_terminal_features_captured_before_reset_without_modifying_original_env(self):
        class Base:
            def reset_idx(self, ids):
                self.root_states[ids, 7] = 99.
                self.dof_pos[ids] = DEFAULT
            def step(self, actions):
                self.root_states[:, 7] = .25
                self.dof_pos += .01
                self.reset_idx(torch.tensor([1]))
                return (None, None, torch.zeros(2), torch.tensor([0, 1]), {}, torch.tensor([1]), None)
        tree = ast.parse((ROOT/'legged_gym/envs/go2_him_amp/amp_env.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name != '__init__']
        scope = dict(torch=torch, Go2HIMFlatQualityRobot=Base)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), '<amp-env-test>', 'exec'), scope)
        env = scope['Go2HIMAMPRobot'](); env.amp_capture_enabled = True; env._amp_collecting_step = False
        env.num_envs, env.device, env.amp_kinematics = 2, 'cpu', self.kin
        env.dof_pos, env.dof_vel = DEFAULT.repeat(2, 1), torch.zeros(2, 12)
        env.root_states = torch.zeros(2, 13); env.root_states[:, 2] = .32; env.root_states[:, 6] = 1
        result = env.step(torch.zeros(2, 12))
        self.assertAlmostEqual(float(result[4]['amp_transition_next'][1, 24]), .25)
        self.assertAlmostEqual(float(env.get_amp_observations()[1, 24]), 99.)
        terminal = result[4]['amp_transition_next'].clone(); env.root_states.zero_()
        torch.testing.assert_close(result[4]['amp_transition_next'], terminal, rtol=0, atol=0)

    def test_existing_amp61_files_are_read_without_retargeting(self):
        with np.load(Path(self.temp.name)/'diagnostic.npz') as original:
            expected = original['features']
        frames = np.zeros((len(expected), 61), dtype=np.float32)
        frames[:, 2], frames[:, 6] = expected[:, -1], 1.
        order = [1, 0, 3, 2]
        for start, source in ((7, 0), (19, 12), (37, 30)):
            frames[:, start:start+12] = expected[:, source:source+12].reshape(-1, 4, 3)[:, order].reshape(-1, 12)
        frames[:, 31:37] = expected[:, 24:30]
        file = Path(self.temp.name)/'original_motion.txt'
        file.write_text(json.dumps(dict(Frames=frames.tolist(), FrameDuration=.02)))
        original_bytes = file.read_bytes()
        manifest = json.loads(self.manifest.read_text())
        manifest['clips'] = [dict(file=file.name, sha256=hashlib.sha256(original_bytes).hexdigest(),
                                  format='amp61', source_leg_order=['FR', 'FL', 'RR', 'RL'])]
        self.manifest.write_text(json.dumps(manifest))
        dataset = MotionDataset(self.manifest, self.kin, allow_diagnostic=True)
        np.testing.assert_array_equal(dataset.clips[0][0].numpy(), expected)
        self.assertEqual(file.read_bytes(), original_bytes)
        manifest['clips'][0].pop('source_leg_order'); self.manifest.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'leg order'):
            MotionDataset(self.manifest, self.kin, allow_diagnostic=True)


if __name__ == '__main__':
    unittest.main()
