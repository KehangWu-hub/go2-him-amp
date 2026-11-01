"""Preserve the approved flat training and bundled inference contract."""
import ast
from pathlib import Path
import unittest
import torch
from tests.test_go2_him_candidate import candidate_scope

ROOT=Path(__file__).resolve().parents[1]

class FlatBoundedTest(unittest.TestCase):
    def test_approved_training_contract(self):
        scope=candidate_scope();p=ROOT/'legged_gym/envs/go2_him/flat_bounded_config.py';t=ast.parse(p.read_text());t.body=[n for n in t.body if isinstance(n,ast.ClassDef)];exec(compile(t,str(p),'exec'),scope)
        cfg,ppo=scope['Go2HIMFlatBoundedCfg'],scope['Go2HIMFlatBoundedCfgPPO']
        self.assertEqual((cfg.env.num_envs,ppo.runner.num_steps_per_env,ppo.runner.max_iterations),(2048,100,3000))
        self.assertEqual(cfg.terrain.mesh_type,'plane');self.assertFalse(cfg.terrain.curriculum)
        self.assertEqual((cfg.normalization.clip_actions,ppo.policy.action_bound),(4.,4.))
        self.assertEqual((ppo.algorithm.schedule,ppo.algorithm.learning_rate),('fixed',.0003))
        self.assertEqual([getattr(cfg.rewards.scales,k) for k in ('dof_vel','torques','joint_pos','dof_pos_limits')],[-.001,-.0002,-.7,-10.])

    def test_bundled_model_matches_full_him_graph_and_bound(self):
        p=ROOT/'deploy/pre_train/go2_him_flat_bounded/policy.pt';m=torch.jit.load(str(p)).eval();g=torch.Generator().manual_seed(19);x=torch.randn(64,270,generator=g)*.2
        with torch.no_grad():
            z=m.estimator(x);a=m.actor(torch.cat((x[:,:45],z[:,:3],torch.nn.functional.normalize(z[:,3:],dim=-1)),dim=-1));expected=4*torch.tanh(a/4);actual=m(x)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        self.assertTrue(bool((actual.abs()<=4).all()))
