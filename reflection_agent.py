import os
import logging
from click import prompt
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, MessagesState
from langgraph_reflection import create_reflection_graph
from langgraph.graph import START, END
from typing import TypedDict
import logging
from pydantic import BaseModel, Field
from sympy.abc import lamda
from utils import parse_pddl_output



class PDDLOutput(BaseModel):
    domain: str = Field(..., description="The PDDL domain file content")
    problem: str = Field(..., description="The PDDL problem file content")

class ReflectionState(TypedDict):
    pddl_domain: str
    pddl_problem: str
    pddl_valid: bool
    last_planner_output: str
    suggested_fix: str
    attempts: int

def run_planner(domain: str, problem: str) -> tuple[str, bool]:
    """
    Run the planner on the given PDDL files and return the output.

    Args:
        domain_path (str): Path to the domain PDDL file.
        problem_path (str): Path to the problem PDDL file.

    Returns:
        (str, Bool): Output from the planner, True if a solution was found, False otherwise.

    """
    import subprocess

    domain_path = 'temp/domain.pddl'
    problem_path = 'temp/problem.pddl'
    if not os.path.isdir('temp'):
        os.mkdir('temp')
    with open(domain_path, 'w') as domain_file, open(problem_path, 'w') as problem_file:
        domain_file.write(domain)
        problem_file.write(problem)
    result = subprocess.run(
        [
            "python", "../planner/fast-downward.py",
            domain_path,
            problem_path,
            "--search", "lazy_greedy([ff()], preferred=[ff()])"
        ],
        capture_output=True,
        text=True
    )

    os.remove(domain_path)
    os.remove(problem_path)
    os.rmdir('temp')
    return result.stdout, 'translate exit code: 0' in result.stdout or int(result.returncode) in [0,1,2,3]

def validate_node(state: ReflectionState):
    """
    Validate the PDDL domain and problem.
    """
    print('='*50 + f"\nReflection Agent - Validating for the {state['attempts']}-th time")
    output, valid = run_planner(state['pddl_domain'], state['pddl_problem'])
    print(f"Reflection Agent - Planner Output: {output}")
    print(f"Reflection Agent - {'No valid solution found' if not valid else 'Valid solution found'}")
    state['last_planner_output'] = output
    state['pddl_valid'] = valid
    return state

def reflect_node(state: ReflectionState):
    advice = reflect_on_failure(state['pddl_domain'], state['pddl_problem'], state['last_planner_output'])
    print('='*50 + f"\nReflection Agent - Suggested Fix: {advice}")
    state['suggested_fix'] = advice
    return state

def apply_fix_node(state: ReflectionState):
    print('='*50 + "\nReflection Agent - Applying Fix")
    response = apply_fix(state['pddl_domain'], state['pddl_problem'], state['suggested_fix'])
    print(f"Reflection Agent - PDDL after the fix:\n{response}")
    new_domain, new_problem = parse_pddl_output(response)

    state['pddl_domain'] = new_domain
    state['pddl_problem'] = new_problem
    state['attempts'] += 1
    print(f"New state:\nAttempts: {state['attempts']}\nPDDL Domain: {state['pddl_domain']}\nPDDL Problem: {state['pddl_problem']}")
    return state

def reflect_on_failure(pddl_domain: str, pddl_problem: str, planner_output: str) -> str:
    llm = ChatOllama(model="gemma3", temperature=0.3)
    prompt = f"""
        You are an assistant that helps a PDDL expert.
        The planner output is:\n{planner_output}\n\n
        The PDDL domain and problem files are as follows:\n
        DOMAIN:\n{pddl_domain}\n\nPROBLEM:\n{pddl_problem}\n\n
        Please analyze the output and suggest improvements to the PDDL files to make them valid or to find a solution. 
        Follow these guidelines:
        1) Focus on errors, not optimizations.
        2) Ignore warnings, they are not relevant.
"""
    return llm.invoke(prompt).content


def apply_fix(pddl_domain: str, pddl_problem: str, suggested_fix: str) -> str:
    llm = ChatOllama(model="qwen2.5-coder:7b", temperature=0.1, num_predict=-1)
    prompt = f"""
    You are a PDDL expert. Given the following PDDL domain and problem, along with a suggested fix, apply the suggested fix to the PDDL files.
FIX:
{suggested_fix}

PDDL domain:
{pddl_domain}

PDDL problem:
{pddl_problem}

Format the output as follows:
<DOMAIN>
pddl domain content here
<PROBLEM>
pddl problem content here

just write the PDDL code, no additional text or explanations except for PDDL comments.
"""
    output = llm.invoke(prompt).content

    return output

def build_reflection_graph(max_attempts: int = 3):
    graph = StateGraph(ReflectionState)
    graph.add_node("validate", RunnableLambda(validate_node))
    graph.add_node("reflect", RunnableLambda(reflect_node))
    graph.add_node("apply_fix", RunnableLambda(apply_fix_node))
    graph.set_entry_point("validate")
    graph.add_conditional_edges(
        "validate",
        path=lambda s: "reflect" if not s["pddl_valid"] and s['attempts'] < max_attempts else END
    )
    graph.add_edge("reflect", "apply_fix")
    graph.add_edge("apply_fix", "validate")
    return graph.compile()
