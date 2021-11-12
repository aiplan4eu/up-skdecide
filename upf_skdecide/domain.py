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

from typing import Dict, Optional, Union
from copy import deepcopy

from skdecide.core import ImplicitSpace, Space, Value
from skdecide.domains import DeterministicPlanningDomain
from skdecide.hub.space.gym import ListSpace

import upf
from upf.substituter import Substituter
from upf.expression import Expression
from upf.plan import ActionInstance
from upf.plan_validator import QuantifierSimplifier, SequentialPlanValidator


class D(DeterministicPlanningDomain):
    T_state = Dict[Expression, Expression]  # Type of states
    T_observation = T_state  # Type of observations
    T_event = ActionInstance  # Type of events
    T_value = float  # Type of transition values (rewards or costs)
    T_info = None  # Type of additional information in environment outcome


class DomainImpl(D, SequentialPlanValidator):
    
    def __init__(self, problem: 'upf.Problem', **options):
        self._problem = problem
        self._env: 'upf.environment.Environment' = upf.get_env(options.get('env', None))
        self._qsimplifier = QuantifierSimplifier(self._env, self._problem)
        self.manager = self._env.expression_manager
        self._substituter = Substituter(self._env)
        self._last_error: Union[str, None] = None
    
    def _get_next_state(self, memory: D.T_state, action: D.T_event) -> D.T_state:
        assert isinstance(action, upf.InstantaneousAction)
        self._count = self._count + 1
        new_assignments: Dict[Expression, Expression] = {}
        for ap, oe in zip(action.action().parameters(), action.actual_parameters()):
            memory[ap] = oe
        for e in action.effects():
            cond = True
            if e.is_conditional():
                ec = self._subs_simplify(e.condition(), memory)
                assert ec.is_bool_constant()
                cond = ec.bool_constant_value()
            if cond:
                ge = self._get_ground_fluent(e.fluent(), memory)
                if e.is_assignment():
                    new_assignments[ge] = self._subs_simplify(e.value(), memory)
                elif e.is_increase():
                    new_assignments[ge] = self._subs_simplify(self.manager.Plus(e.fluent(),
                                            e.value()), memory)
                elif e.is_decrease():
                    new_assignments[ge] = self._subs_simplify(self.manager.Minus(e.fluent(),
                                            e.value()), memory)
        next_state = deepcopy(memory)
        next_state.update(new_assignments)
        for ap in action.parameters():
            del next_state[ap]
        return next_state
    
    def _get_transition_value(self, memory: D.T_state, action: D.T_event, next_state: Optional[D.T_state] = None) -> Value[D.T_value]:
        # TODO: how to get the reward or cost fluent?
        pass
    
    def _is_terminal(self, state: D.T_state) -> D.T_predicate:
        for g in self._problem.goals():
            gs = self._subs_simplify(g, state)
            if not (gs.is_bool_constant() and gs.bool_constant_value()):
                return False
        return True
    
    def _get_action_space_(self) -> Space[D.T_event]:
        # TODO: how to get all the action instantiations? Do we really want to do that?
        return ListSpace([ActionInstance()])
    
    def _get_applicable_actions_from(self, memory: D.T_state) -> Space[D.T_event]:
        actions = []
        for ai in self._problem.actions:
            for p in ai.preconditions():
                ps = self._subs_simplify(p, memory)
                if ps.is_bool_constant() and ps.bool_constant_value():
                    actions.append(ai)
        # TODO: how to get action instantiations?
        return ListSpace(actions)
    
    def _get_goals_(self) -> Space[D.T_observation]:
        return ImplicitSpace(lambda s: self._is_terminal(s))
    
    def _get_initial_state_(self) -> D.T_state:
        self._last_error = None
        self._count = 0
        return self._problem.initial_values().copy()
    
    def _get_observation_space_(self) -> Space[D.T_observation]:
        # TODO: not clear what to do here, it will depend on the algorithm
        pass