import unified_planning
from unified_planning.shortcuts import *
from unified_planning.test import TestCase, main, skipIfSolverNotAvailable
from unified_planning.test.examples import get_example_problems

from up_skdecide.domain import DomainImpl as SkDecideDomain


class TestSkDecidePlanner(TestCase):
    def setUp(self):
        TestCase.setUp(self)
        self.problems = get_example_problems()

    def test_domain_basic(self):
        problem = self.problems["basic"].problem
        domain = SkDecideDomain(problem)
        domain._get_next_state(
            domain._get_initial_state(), domain._get_action_space().sample()
        )
