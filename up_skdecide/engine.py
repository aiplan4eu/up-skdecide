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

from copy import deepcopy
import inspect
import unified_planning as up
from unified_planning.engines import PlanGenerationResultStatus, Engine, Credits
from unified_planning.engines.mixins import OneshotPlannerMixin
from unified_planning.plans import ActionInstance, SequentialPlan
from unified_planning.model import ProblemKind
from typing import Optional, Callable, IO
from skdecide.hub.domain.up import UPDomain
from skdecide.solvers import Solver as SkDecideSolver
from skdecide.utils import match_solvers
from skdecide.hub.solver.iw import IW


credits = Credits(
    "Scikit-decide",
    "Airbus AI Research",
    "scikit-decide@airbus.com",
    "https://airbus.github.io/scikit-decide",
    "MIT",
    "Scikit-decide is an AI framework for Reinforcement Learning, Automated Planning and Scheduling.",
    "Scikit-decide is an AI framework for Reinforcement Learning, Automated Planning and Scheduling.",
)


class EngineImpl(Engine, OneshotPlannerMixin):
    """Represents the engine interface."""

    def __init__(self, **options):
        super().__init__()
        self.optimality_metric_required = False
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
        self._simulator_params = (
            self._options["simulator_params"]
            if "simulator_params" in self._options
            else dict()
        )

    @property
    def name(self) -> str:
        return "SkDecide"

    @staticmethod
    def get_credits(**kwargs) -> Optional["Credits"]:
        return credits

    @staticmethod
    def supported_kind() -> "ProblemKind":
        supported_kind = ProblemKind()
        supported_kind.set_problem_type("SIMPLE_NUMERIC_PLANNING")
        supported_kind.set_problem_class("ACTION_BASED")
        supported_kind.set_time("DISCRETE_TIME")
        supported_kind.set_parameters("BOOL_FLUENT_PARAMETERS")
        supported_kind.set_parameters("BOUNDED_INT_FLUENT_PARAMETERS")
        supported_kind.set_parameters("BOOL_ACTION_PARAMETERS")
        supported_kind.set_parameters("BOUNDED_INT_ACTION_PARAMETERS")
        supported_kind.set_parameters("UNBOUNDED_INT_ACTION_PARAMETERS")
        supported_kind.set_numbers("DISCRETE_NUMBERS")
        supported_kind.set_numbers("BOUNDED_TYPES")
        supported_kind.set_typing("HIERARCHICAL_TYPING")
        supported_kind.set_fluents_type("INT_FLUENTS")
        supported_kind.set_fluents_type("REAL_FLUENTS")
        supported_kind.set_fluents_type("OBJECT_FLUENTS")
        supported_kind.set_conditions_kind("NEGATIVE_CONDITIONS")
        supported_kind.set_conditions_kind("DISJUNCTIVE_CONDITIONS")
        supported_kind.set_conditions_kind("EQUALITIES")
        supported_kind.set_conditions_kind("EXISTENTIAL_CONDITIONS")
        supported_kind.set_conditions_kind("UNIVERSAL_CONDITIONS")
        supported_kind.set_simulated_entities("SIMULATED_EFFECTS")
        supported_kind.set_effects_kind("INCREASE_EFFECTS")
        supported_kind.set_effects_kind("DECREASE_EFFECTS")
        supported_kind.set_effects_kind("CONDITIONAL_EFFECTS")
        supported_kind.set_effects_kind("STATIC_FLUENTS_IN_BOOLEAN_ASSIGNMENTS")
        supported_kind.set_effects_kind("STATIC_FLUENTS_IN_NUMERIC_ASSIGNMENTS")
        supported_kind.set_effects_kind("STATIC_FLUENTS_IN_OBJECT_ASSIGNMENTS")
        supported_kind.set_effects_kind("FLUENTS_IN_BOOLEAN_ASSIGNMENTS")
        supported_kind.set_effects_kind("FLUENTS_IN_NUMERIC_ASSIGNMENTS")
        supported_kind.set_effects_kind("FLUENTS_IN_OBJECT_ASSIGNMENTS")
        supported_kind.set_effects_kind("FORALL_EFFECTS")
        supported_kind.set_actions_cost_kind("STATIC_FLUENTS_IN_ACTIONS_COST")
        supported_kind.set_actions_cost_kind("FLUENTS_IN_ACTIONS_COST")
        supported_kind.set_actions_cost_kind("INT_NUMBERS_IN_ACTIONS_COST")
        supported_kind.set_actions_cost_kind("REAL_NUMBERS_IN_ACTIONS_COST")
        supported_kind.set_simulated_entities("SIMULATED_EFFECTS")
        supported_kind.set_quality_metrics("ACTIONS_COST")
        supported_kind.set_quality_metrics("FINAL_VALUE")
        supported_kind.set_quality_metrics("PLAN_LENGTH")
        return supported_kind

    @staticmethod
    def supports(problem_kind: "ProblemKind") -> bool:
        return problem_kind <= EngineImpl.supported_kind()

    def _solve(
        self,
        problem: "up.model.Problem",
        callback: Optional[
            Callable[["up.engines.results.PlanGenerationResult"], None]
        ] = None,
        timeout: Optional[float] = None,
        output_stream: Optional[IO[str]] = None,
    ) -> "up.engines.PlanGenerationResultStatus":
        solver_config = deepcopy(self._solver_config)
        if "fluent_domains" in solver_config:
            fluent_domains = solver_config["fluent_domains"]
            del solver_config["fluent_domains"]
        else:
            fluent_domains = None
        if "state_encoding" in solver_config:
            state_encoding = solver_config["state_encoding"]
            del solver_config["state_encoding"]
        else:
            state_encoding = "native"
        if "action_encoding" in solver_config:
            action_encoding = solver_config["action_encoding"]
            del solver_config["action_encoding"]
        else:
            action_encoding = "native"
        domain_factory = lambda: UPDomain(
            problem,
            fluent_domains=fluent_domains,
            state_encoding=state_encoding,
            action_encoding=action_encoding,
            **self._simulator_params
        )
        domain = domain_factory()
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
            solver_config.update({"domain_factory": domain_factory})
        plan = []
        with self._solver_class(**solver_config) as solver:
            solver.solve()
            rollout_domain = domain_factory()
            state = rollout_domain.reset()
            while not rollout_domain.is_terminal(state):
                action = solver.sample_action(
                    state
                )  # TODO: handle when no action is applicable
                state = rollout_domain.get_next_state(state, action)
                plan.append(rollout_domain._convert_to_skup_action_(action))
        seq_plan = SequentialPlan([ActionInstance(x._up_action) for x in plan])
        return up.engines.PlanGenerationResult(
            PlanGenerationResultStatus.SOLVED_SATISFICING, seq_plan, self.name
        )
