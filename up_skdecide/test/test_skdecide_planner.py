import unified_planning
from unified_planning.shortcuts import *
from unified_planning.test import TestCase, main, skipIfEngineNotAvailable
from unified_planning.test.examples import get_example_problems

from up_skdecide.domain import DomainImpl as SkDecideDomain

from skdecide.hub.solver.iw import IW


class TestSkDecidePlanner(TestCase):
    def setUp(self):
        TestCase.setUp(self)
        if "skdecide" not in get_env().factory.engines:
            get_env().factory.add_engine("skdecide", "up_skdecide", "EngineImpl")
        self.problems = get_example_problems()

    def test_domain_basic(self):
        problem = self.problems["basic"].problem
        domain = SkDecideDomain(problem)
        domain._get_next_state(
            domain._get_initial_state(), domain._get_action_space().sample()
        )

    def test_planner_basic(self):
        problem, plan = self.problems["basic"]
        with OneshotPlanner(
            name="skdecide",
            params={
                "solver": IW,
                "config": {"state_features": lambda d, s: s},
            },
        ) as planner:
            self.assertNotEqual(planner, None)
            res = planner.solve(problem)
            self.assertEqual(str(plan), str(res.plan))
