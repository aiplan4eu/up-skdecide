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
"""This module defines the skdecide domain interface."""

from typing import Dict, Optional

from skdecide.core import ImplicitSpace, Space, Value
from skdecide.domains import DeterministicPlanningDomain
from skdecide.hub.space.gym import ListSpace, MultiBinarySpace

import unified_planning as up
from unified_planning.shortcuts import *


class D(DeterministicPlanningDomain):
    T_state = Dict[up.model.Expression, up.model.Expression]  # Type of states
    T_observation = T_state  # Type of observations
    T_event = up.model.InstantaneousAction  # Type of events
    T_value = float  # Type of transition values (rewards or costs)
    T_info = None  # Type of additional information in environment outcome


class DomainImpl(D, up.solvers.plan_validator.SequentialPlanValidator):
    def __init__(self, problem: "up.model.Problem", **options):
        up.solvers.plan_validator.SequentialPlanValidator.__init__(self, **options)
        self._problem = problem
        self._env: "up.environment.Environment" = up.environment.get_env()
        with self._env.factory.Grounder(
            problem_kind=up.model.ProblemKind()
        ) as grounder:
            self._grounded_problem, self._rewrite_back_plan_function = grounder.ground(
                problem
            )
        self._qsimplifier = up.solvers.plan_validator.QuantifierSimplifier(
            self._env, self._grounded_problem
        )
        self._initial_state_dict = self._grounded_problem.initial_values().copy()
        self._state_dict_keys = self._initial_state_dict.keys()

    @property
    def grounded_problem(self) -> str:
        """Returns the grounded problem."""
        return self._grounded_problem

    @property
    def rewrite_back_plan_function(self) -> str:
        """Returns the back plan rewriter."""
        return self._rewrite_back_plan_function

    def _get_next_state(self, memory: D.T_state, action: D.T_event) -> D.T_state:
        assert isinstance(action, up.model.InstantaneousAction)
        assignments = {k: memory[i] for i, k in enumerate(self._state_dict_keys)}
        new_assignments: Dict[up.model.Expression, up.model.Expression] = {}
        for ap, oe in zip(action.parameters(), action.parameters()):
            assignments[ap] = oe
        for p in action.preconditions():
            ps = self._subs_simplify(p, assignments)
            if not (ps.is_bool_constant() and ps.bool_constant_value()):
                self._last_error = (
                    f"Precondition {p} of action {str(action)} is not satisfied."
                )
                return False
        for e in action.effects():
            cond = True
            if e.is_conditional():
                ec = self._subs_simplify(e.condition(), assignments)
                assert ec.is_bool_constant()
                cond = ec.bool_constant_value()
            if cond:
                ge = self._get_ground_fluent(e.fluent(), assignments)
                if e.is_assignment():
                    new_assignments[ge] = self._subs_simplify(e.value(), assignments)
                elif e.is_increase():
                    new_assignments[ge] = self._subs_simplify(
                        self.manager.Plus(e.fluent(), e.value()), assignments
                    )
                elif e.is_decrease():
                    new_assignments[ge] = self._subs_simplify(
                        self.manager.Minus(e.fluent(), e.value()), assignments
                    )
        assignments.update(new_assignments)
        for ap in action.parameters():
            del assignments[ap]
        return list(assignments.values())

    def _get_transition_value(
        self,
        memory: D.T_state,
        action: D.T_event,
        next_state: Optional[D.T_state] = None,
    ) -> Value[D.T_value]:
        # TODO: how to get the reward or cost fluent?
        return Value(cost=1)

    def _is_terminal(self, memory: D.T_state) -> D.T_predicate:
        state = {k: memory[i] for i, k in enumerate(self._state_dict_keys)}
        for g in self._grounded_problem.goals():
            gs = self._subs_simplify(g, state)
            if not (gs.is_bool_constant() and gs.bool_constant_value()):
                return False
        return True

    def _get_action_space_(self) -> Space[D.T_event]:
        # TODO: how to get all the action instantiations? Do we really want to do that?
        return ListSpace(self._grounded_problem.actions())

    def _get_applicable_actions_from(self, memory: D.T_state) -> Space[D.T_event]:
        state = {k: memory[i] for i, k in enumerate(self._state_dict_keys)}
        actions = []
        for ai in self._grounded_problem.actions():
            for p in ai.preconditions():
                ps = self._subs_simplify(p, state)
                if ps.is_bool_constant() and ps.bool_constant_value():
                    actions.append(ai)
        return ListSpace(actions)

    def _get_goals_(self) -> Space[D.T_observation]:
        return ImplicitSpace(lambda s: self._is_terminal(s))

    def _get_initial_state_(self) -> D.T_state:
        self._last_error = None
        return list(self._initial_state_dict.values())

    def _get_observation_space_(self) -> Space[D.T_observation]:
        # TODO: not clear what to do here, it will depend on the algorithm
        return MultiBinarySpace(len(self._initial_state_dict))
