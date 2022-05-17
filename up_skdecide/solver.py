# Copyright 2021 AIPlan4EU project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""This module defines the solver interface."""

import inspect
import unified_planning as up
from unified_planning.solvers import PlanGenerationResultStatus
from unified_planning.plan import ActionInstance, SequentialPlan
from unified_planning.model import ProblemKind
from typing import Optional, Tuple, Callable, Union
from skdecide.solvers import Solver as SkDecideSolver
from skdecide.utils import match_solvers
from .domain import DomainImpl


class SolverImpl(up.solvers.Solver):
    """Represents the solver interface."""

    def __init__(self, **options):
        if (
            len(options) != 2
            or "solver" not in options
            or not issubclass(options["solver"], SkDecideSolver)
            or "config" not in options
            or not isinstance(options["config"], dict)
        ):
            raise RuntimeError(
                "SkDecide's UP solver only accepts the 'solver' option (SkDecide's underlying solver) and its config dictionary. Provided options: {}".format(
                    options
                )
            )
        self._solver_class = options["solver"]
        self._solver_config = options["config"]
        self._options = options

    @property
    def name(self) -> str:
        return "SkDecide"

    @staticmethod
    def is_oneshot_planner() -> bool:
        return True

    @staticmethod
    def satisfies(optimality_guarantee: Union[int, str]) -> bool:
        return False

    @staticmethod
    def supports(problem_kind: "ProblemKind") -> bool:
        supported_kind = ProblemKind()
        # supported_kind.set_time("DISCRETE_TIME") // This is not supported at the moment
        return problem_kind.features().issubset(supported_kind.features())

    def solve(self, problem: "up.model.Problem") -> "up.solvers.PlanGenerationResultStatus":
        domain = DomainImpl(problem)
        if len(match_solvers(domain, [self._solver_class])) == 0:
            raise RuntimeError(
                "The scikit-decide's solver {} is not compatible with this problem".format(
                    self._solver_class.__name__
                )
            )
        if (
            "domain_factory"
            in inspect.signature(self._solver_class.__init__).parameters
        ):
            self._solver_config.update(
                {"domain_factory": lambda: DomainImpl(problem, **self._options)}
            )
        plan = []
        with self._solver_class(**self._solver_config) as solver:
            solver.solve(lambda: DomainImpl(problem, **self._options))
            rollout_domain = DomainImpl(problem, **self._options)
            state = rollout_domain.reset()
            while not rollout_domain.is_terminal(state):
                action = solver.sample_action(state)
                state = rollout_domain.get_next_state(state, action)
                plan.append(rollout_domain.grounded_problem.action(action.name))
        seq_plan = rollout_domain.rewrite_back_plan(
            SequentialPlan([ActionInstance(x) for x in plan])
        )
        return up.solvers.PlanGenerationResult(PlanGenerationResultStatus.SOLVED_SATISFICING, seq_plan, self.name)

    def destroy(self):
        pass
