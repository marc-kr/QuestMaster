import hashlib
import json
from langchain_ollama import ChatOllama
from pddlpy import DomainProblem

import typing

from pddlpy.pddl import Operator

def state_hash(state: set) -> str:
    sorted_facts = sorted([f"{pred}({','.join(args)})" for pred, *args in state])
    serialized_state = ';'.join(sorted_facts)
    return hashlib.md5(serialized_state.encode()).hexdigest()

def available_actions(dp, operators, state) -> typing.List[Operator]:
    """
    Returns a list of actions that can be applied in the given state.
    """
    actions = []
    for operator in operators:
        for action in dp.ground_operator(operator):
            if action.precondition_pos.issubset(state) and action.precondition_neg.isdisjoint(state):
                actions.append(action)
    return actions

def take_action(action: Operator, state: set) -> set:
    """
    Applies the given action to the state and returns the new state.
    """
    new_state = state.copy()
    new_state = new_state.difference(action.effect_neg).union(action.effect_pos)
    return new_state

def is_applicable(action: Operator, state: set) -> bool:
    """
    Checks if the action can be applied in the given state.
    """
    return (action.precondition_pos.issubset(state) and
            action.precondition_neg.isdisjoint(state))

def compute_states(dp: DomainProblem):
    initial_state = set([tuple(atom.predicate) for atom in dp.initialstate()])
    goal = set([tuple(atom.predicate) for atom in dp.goals()])
    operators = dp.operators()

    visited = set()
    frontier = [initial_state]
    transitions = {}
    while frontier:
        state = frontier.pop()
        state_id = state_hash(state)
        if state_id in visited:
            continue
        visited.add(state_id)
        applicable_actions = available_actions(dp, operators, state)
        transitions[state_id] = {}
        transitions[state_id]['state'] = state
        transitions[state_id]['actions'] = []

        for action in applicable_actions:

            new_state = take_action(action, state)
            is_goal = goal.issubset(new_state)
            #print('action ', action.variable_list)
            transitions[state_id]['actions'].append({
                'action': f"{action.operator_name} {' '.join(action.variable_list.values())}",
                'new_state': {'state_id': 'goal' if is_goal else state_hash(new_state), 'state': new_state},
            })
            if not is_goal and state_hash(new_state) not in visited:
                frontier.append(new_state)
    return transitions

def compute_states_transitions(domain_path: str, problem_path: str):
    dp = DomainProblem(domain_path, problem_path)
    init_state = set([tuple(atom.predicate) for atom in dp.initialstate()])
    goal = set([tuple(atom.predicate) for atom in dp.goals()])

    states_transitions = compute_states(dp)

    for t in states_transitions:
        states_transitions[t]['state'] = list(states_transitions[t]['state'])
        for action in states_transitions[t]['actions']:
            action['new_state']['state'] = list(action['new_state']['state'])
    return states_transitions

if __name__ == "__main__":
    domain = './testing/pddl_tests/save-the-galactic-ambassador-domain.pddl'
    problem = './testing/pddl_tests/save-the-galactic-ambassador-problem.pddl'
    #domain = 'gen_domain.pddl'
    #problem = 'gen_problem.pddl'
    #domain = 'sga-domain-fixed.pddl'
    #problem = 'sga-problem-fixed.pddl'

    states_transitions = compute_states_transitions(domain, problem)

    print(states_transitions)
    print(json.dumps(states_transitions, indent=2))