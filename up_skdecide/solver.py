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

import unified_planning as up
import unified_planning.model
from unified_planning.plan import (
    Plan,
    ActionInstance,
    SequentialPlan,
    TimeTriggeredPlan,
)
from unified_planning.model import ProblemKind, Problem, Action, FNode
from functools import partial
from typing import Optional, Tuple, Dict, List, Callable, Union
from skdecide.solvers import Solver as SkDecideSolver
from skdecide.utils import match_solvers
from .domain import DomainImpl


OPTIMALITY_GUARANTEES = list(range(0, 2))

(SATISFICING, OPTIMAL) = OPTIMALITY_GUARANTEES


class SolverImpl(up.solvers.Solver):
    """Represents the solver interface."""

    def __init__(self, **options):
        if (
            len(options) != 2
            or "solver" not in options
            or not isinstance(options["solver"], SkDecideSolver)
        ):
            raise RuntimeError(
                "SkDecide's UP solver only accepts the 'solver' option (SkDecide's underlying solver) and its config dictionary"
            )
        self._solver = options["solver"]

    @staticmethod
    def name() -> str:
        return "SkDecide"

    @staticmethod
    def is_oneshot_planner() -> bool:
        return True

    @staticmethod
    def satisfies(optimality_guarantee: Union[int, str]) -> bool:
        return False

    @staticmethod
    def is_plan_validator() -> bool:
        return False

    @staticmethod
    def is_grounder() -> bool:
        return False

    @staticmethod
    def supports(problem_kind: "ProblemKind") -> bool:
        supported_kind = ProblemKind()
        supported_kind.set_time("DISCRETE_TIME")
        return problem_kind.features().issubset(supported_kind.features())

    def solve(self, problem: "up.model.Problem") -> Optional["up.plan.Plan"]:
        domain = DomainImpl(problem)
        if len(match_solvers(domain, [self._solver])) == 0:
            raise RuntimeError(
                "The scikit-decide's solver {} is not compatible with this problem".format(
                    self._solver.__name__
                )
            )
        # with self._solver()

    def validate(self, problem: "up.model.Problem", plan: "up.plan.Plan") -> bool:
        raise NotImplementedError

    def ground(
        self, problem: "up.model.Problem"
    ) -> Tuple[Problem, Callable[[Plan], Plan]]:
        """
        Implement only if "self.is_grounder()" returns True.
        This function should return the tuple (grounded_problem, trace_back_plan), where
        "trace_back_plan" is a callable from a plan for the "grounded_problem" to a plan of the
        original problem.

        NOTE: to create a callable, the "functools.partial" method can be used, as we do in the
        "up.solvers.grounder".

        Also, the "up.solvers.grounder.lift_plan" function can be called, if retrieving the needed map
        fits the solver implementation better than retrieving a function."""
        raise NotImplementedError

    def destroy(self):
        raise NotImplementedError

    def __enter__(self):
        """Manages entering a Context (i.e., with statement)"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Manages exiting from Context (i.e., with statement)"""
        self.destroy()
