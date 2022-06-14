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
"""This module defines the engine interface."""

import inspect
import unified_planning as up
from unified_planning.engines import PlanGenerationResultStatus, Engine, Credits
from unified_planning.engines.mixins import OneshotPlannerMixin
from unified_planning.plans import ActionInstance, SequentialPlan
from unified_planning.model import ProblemKind
from typing import Optional, Tuple, Callable, Union
from skdecide.solvers import Solver as SkDecideSolver
from skdecide.utils import match_solvers
from skdecide.hub.solver.iw import IW
from .domain import DomainImpl


credits = Credits('Scikit-decide',
                  'Airbus Team',
                  'florent.teichteil-koenigsbuch@airbus.com',
                  'https://airbus.github.io/scikit-decide',
                  'MIT',
                  'Scikit-decide is an AI framework for Reinforcement Learning, Automated Planning and Scheduling.',
                  'Scikit-decide is an AI framework for Reinforcement Learning, Automated Planning and Scheduling.')


class EngineImpl(Engine, OneshotPlannerMixin):
    """Represents the engine interface."""

    def __init__(self, **options):
        if len(options) == 0:
            self._options = {
                "solver": IW,
                "config": {"state_features": lambda d, s: s},
            }
        elif (
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
        else:
            self._options = options
        self._solver_class = self._options["solver"]
        self._solver_config = self._options["config"]

    @property
    def name(self) -> str:
        return "SkDecide"

    @staticmethod
    def get_credits(**kwargs) -> Optional["Credits"]:
        return credits

    @staticmethod
    def supported_kind() -> "ProblemKind":
        supported_kind = ProblemKind()
        supported_kind.set_problem_class("ACTION_BASED")
        supported_kind.set_numbers("DISCRETE_NUMBERS")
        supported_kind.set_numbers("CONTINUOUS_NUMBERS")
        supported_kind.set_typing("FLAT_TYPING")
        supported_kind.set_conditions_kind("NEGATIVE_CONDITIONS")
        supported_kind.set_conditions_kind("DISJUNCTIVE_CONDITIONS")
        supported_kind.set_conditions_kind("EQUALITY")
        supported_kind.set_conditions_kind("EXISTENTIAL_CONDITIONS")
        supported_kind.set_conditions_kind("UNIVERSAL_CONDITIONS")
        supported_kind.set_fluents_type("NUMERIC_FLUENTS")
        supported_kind.set_fluents_type("OBJECT_FLUENTS")
        supported_kind.set_simulated_entities("SIMULATED_EFFECTS")
        supported_kind.set_effects_kind("INCREASE_EFFECTS")
        supported_kind.set_effects_kind("DECREASE_EFFECTS")
        supported_kind.set_effects_kind("CONDITIONAL_EFFECTS")
        return supported_kind

    @staticmethod
    def supports(problem_kind: "ProblemKind") -> bool:
        return problem_kind <= EngineImpl.supported_kind()

    def solve(
        self, problem: "up.model.Problem"
    ) -> "up.engines.PlanGenerationResultStatus":
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
                action = solver.sample_action(state) # TODO: handle when no action is applicable
                state = rollout_domain.get_next_state(state, action)
                plan.append(rollout_domain.grounded_problem.action(action.name))
        seq_plan = rollout_domain.rewrite_back_plan(
            SequentialPlan([ActionInstance(x) for x in plan])
        )
        return up.engines.PlanGenerationResult(
            PlanGenerationResultStatus.SOLVED_SATISFICING, seq_plan, self.name
        )
