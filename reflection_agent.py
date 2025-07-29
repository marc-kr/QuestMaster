import os

from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph

from langgraph.graph import END
from typing import TypedDict, Callable

from pydantic import BaseModel, Field

from pddl_utils import compute_states_transitions

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage

FAST_DOWNWARD_PATH = ''

FAST_DOWNWARDS_EXIT_CODES = {
    0: "All run components successfully terminated (translator: completed, search: found a plan, validate: validated a plan)",
    1: "Only returned by portfolios: at least one plan was found and another component ran out of memory.",
    2: "Only returned by portfolios: at least one plan was found and another component ran out of time.",
    3: "Only returned by portfolios: at least one plan was found, another component ran out of memory, and yet another one ran out of time.",
    10: "Translator proved task to be unsolvable. Currently not used",
    11: "Task is provably unsolvable with current bound. Currently only used by hillclimbing search.",
    12: "Search ended without finding a solution.",
    20: "Memory exhausted.",
    21: "Time exhausted. Not supported on Windows because we use SIGXCPU to kill the planner.",
    22: "Memory exhausted.",
    23: "Timeout occurred. Not supported on Windows because we use SIGXCPU to kill the planner.",
    24: "Only returned by portfolios: one component ran out of memory and another one out of time.",
    30: "Critical error: something went wrong (e.g. translator bug, but also malformed PDDL input).",
    31: "Usage error: wrong command line options or invalid PDDL inputs",
    32: "Something went wrong that should not have gone wrong (e.g. planner bug).",
    33: "Wrong command line options or SAS+ file.",
    34: "Requested unsupported feature.",
    35: "Something went wrong in the driver (e.g. failed setting resource limits, ill-defined portfolio, complete plan generated after an incomplete one).",
    36: "Usage error: wrong or missing command line options, including (implicitly) specifying non-existing paths (e.g. for input files or build directory).",
    37: "Requested unsupported feature (e.g. limiting memory on macOS)."
}

class PDDLOutput(BaseModel):
    domain_code: str = Field(..., description="The PDDL domain code")
    problem_code: str = Field(..., description="The PDDL problem code")

class ReflectionState(TypedDict):
    pddl_domain: str
    pddl_problem: str
    pddl_valid: bool
    last_planner_output: str
    suggested_fix: str
    attempts: int
    user_feedback: str
    plan: list[str]


class PDDLReflectionAgent:
    def __init__(self,
                 user_input_callback: Callable[[str], str],
                 success_callback: Callable[[list[str]], None] = None,
                 pddl_update_callback: Callable[[str, str], None] = None,
                 planner_update_callback: Callable[[str], None] = None,
                 coder_llm = 'qwen2.5-coder:7b',
                 reflection_llm = 'deepseek-r1:7b',
                 max_attempts: int = 3):
        """
        Initialize the PDDL Reflection Agent with callbacks for success, user input, and PDDL updates.

        Args:
            success_callback (Callable): Callback to call when a valid solution is found.
            user_input_callback (Callable): Callback to get user input for applying fixes.
            pddl_update_callback (Callable): Callback to update the PDDL domain and problem.
        """
        self.success_callback = success_callback if success_callback else lambda plan: None
        self.user_input_callback = user_input_callback
        self.pddl_update_callback = pddl_update_callback if pddl_update_callback else lambda domain, problem: None
        self.planner_update_callback = planner_update_callback if planner_update_callback else lambda output: None
        self.reflection_graph = self.build_reflection_graph(max_attempts)

    def validate_node(self, state: ReflectionState):
        """
        Validate the PDDL domain and problem.
        """
        print('=' * 50 + f"\nReflection Agent - Validating for the {state['attempts']}-th time")
        output, valid, plan = run_planner(state['pddl_domain'], state['pddl_problem'])
        print(f"Reflection Agent - Planner Output: {output}")
        print(f"Reflection Agent - {'No valid solution found' if not valid else 'Valid solution found'}")
        state['last_planner_output'] = output
        state['pddl_valid'] = valid
        state['plan'] = plan
        self.planner_update_callback(output)
        if valid:
            self.success_callback(plan)
        return state

    def reflect_node(self, state: ReflectionState):
        print("Reflection Agent - No valid solution found, reflecting on the failure")
        advice = reflect_on_failure(state['pddl_domain'], state['pddl_problem'], state['last_planner_output'])
        #print('=' * 50 + f"\nReflection Agent - Suggested Fix: {advice}")
        state['suggested_fix'] = advice

        return state

    def user_input(self, msg):
        user_input_msg = self.user_input_callback(msg)
        return user_input_msg

    def user_input_node(self, state: ReflectionState):
        print("Waiting for user input to apply the suggested fix")
        state['user_feedback'] = self.user_input_callback(state['suggested_fix'])
        print(f"Reflection Agent - User feedback: {state['user_feedback']}")
        return state

    def suggestion_refinement_node(self, state: ReflectionState):
        llm = ChatOllama(model='gemma3', temperature=.1)
        prompt = f"""Given the following AI message:\n{state['suggested_fix']}\n\nAnd the user feedback:\n{state['user_feedback']}\n\nDecide whether the user accepts the suggested fix or want to refine it based on its feedback. Expected output:\n'accept' or 'refine'."""
        response = llm.invoke(prompt).content.strip().lower()
        print("Reflection Agent - User response to suggestion:", response)

        if response == 'accept':
            state['user_feedback'] = ''
            state['chat_history'] = [SystemMessage(content='You are an assistant helping a PDDL expert. Basically, you are the navigator in a pair programming approach. You will analyze PDDL code and issues stated by the planner and eventually refine your suggestions based on user feedback.')]
        return state

    def elaborate_user_feedback_node(self, state: ReflectionState):
        print("Regenerating suggestion based on user feedback...")
        prompt = f"""Based on the following AI message:\n{state['suggested_fix']}\n\nAnd the user feedback:\n{state['user_feedback']}\n\nPlease refine your suggestion based on the user feedback. Be clear and concise in your suggestions, as they will be used by a PDDL expert to fix the code. Don't output the whole PDDL code, the driver will handle that."""
        llm = ChatOllama(model="qwen2.5-coder:7b", temperature=0.1, num_predict=-1)
        refined_suggestion = llm.invoke(prompt).content
        print(f"Reflection Agent - New suggestion based on user feedback: {refined_suggestion}")
        state['suggested_fix'] = refined_suggestion

        return state

    def apply_fix_node(self, state: ReflectionState):
        print('=' * 50 + "\nReflection Agent - Applying Fix")
        response = apply_fix(state['pddl_domain'], state['pddl_problem'], state['suggested_fix'])
        #print(f"Reflection Agent - PDDL after the fix:\n{response}")
        new_domain, new_problem = response.domain_code, response.problem_code

        state['pddl_domain'] = new_domain
        state['pddl_problem'] = new_problem
        state['attempts'] += 1
        self.pddl_update_callback(state['pddl_domain'], state['pddl_problem'])
        return state

    def build_reflection_graph(self, max_attempts: int = 3):
        graph = StateGraph(ReflectionState)
        graph.add_node("validate", RunnableLambda(self.validate_node))
        graph.add_node("reflect", RunnableLambda(self.reflect_node))
        graph.add_node("apply_fix", RunnableLambda(self.apply_fix_node))
        graph.add_node("user_input", RunnableLambda(self.user_input_node))
        graph.add_node("suggestion_refinement_router", RunnableLambda(self.suggestion_refinement_node))
        graph.add_node("elaborate_user_feedback", RunnableLambda(self.elaborate_user_feedback_node))
        graph.set_entry_point("validate")
        graph.add_conditional_edges(
            "validate",
            path=lambda s: "reflect" if not s["pddl_valid"] and s['attempts'] < max_attempts else END
        )
        graph.add_edge("reflect", "user_input")
        graph.add_edge("user_input", "suggestion_refinement_router")
        graph.add_conditional_edges("suggestion_refinement_router",
                                    path=lambda s: "elaborate_user_feedback" if s['user_feedback'] else "apply_fix")
        #graph.add_edge('user_input', 'elaborate_user_feedback')
        #graph.add_edge('elaborate_user_feedback', 'apply_fix')
        graph.add_edge("elaborate_user_feedback", "user_input")
        graph.add_edge("apply_fix", "validate")
        return graph.compile()

    def invoke(self, pddl_domain: str, pddl_problem: str):
        """
        Invoke the reflection agent with the given PDDL domain and problem.

        Args:
            pddl_domain (str): The PDDL domain file content.
            pddl_problem (str): The PDDL problem file content.

        Returns:
            PDDLOutput: The output containing the updated PDDL domain and problem.
        """
        state = ReflectionState(
            pddl_domain=pddl_domain,
            pddl_problem=pddl_problem,
            pddl_valid=False,
            last_planner_output="",
            suggested_fix="",
            attempts=0,
            user_feedback="",
            plan=[],
        )
        print(f"Reflection Agent - Initial PDDL:\nDomain: {state['pddl_domain']}\nProblem: {state['pddl_problem']}")
        result = self.reflection_graph.invoke(state)
        return result['pddl_domain'], result['pddl_problem']


def run_planner(domain: str, problem: str) -> tuple[str, bool, list[str]]:
    """
    Run the planner on the given PDDL files and return the output.

    Args:
        :param problem:
        :param domain:

    Returns:
        (str, Bool, list[str]): Output from the planner, True if a solution was found, False otherwise, list of actions to reach the goal.
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
            "python", "./planner/fast-downward.py",
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
    solution_found = False
    plan = []
    if os.path.isfile('sas_plan'):
        solution_found = True
        with open('sas_plan', 'r') as plan_file:
            plan = [line.strip() for line in plan_file][:-1]
    print(f"Planner Output in Run Planner:\n{result.stdout}")
    print(f"Planner Error: {result.stderr}")
    return (f"{result.stdout}\n**EXIT CODE {result.returncode}: {FAST_DOWNWARDS_EXIT_CODES[result.returncode] 
    if result.returncode in FAST_DOWNWARDS_EXIT_CODES.keys() 
    else ''}**",
            solution_found,
            plan)


def reflect_on_failure(pddl_domain: str, pddl_problem: str, planner_output: str) -> str:
    llm = ChatOllama(model="qwen2.5-coder:7b", temperature=0.1)
    prompt = f"""Analyze the following PDDL planner output and extract the information that is strictly needed to address the found issues.
Ignore all info messages and focus on errors and warnings.\n
Please focus on exit code, here are the meanings of the exit codes:
    - 10 to 12: The goal is not reachable though the PDDL code is syntactically correct. This suggests that there are some issues in the logic of the PDDL code.
    - 30 to 37: The PDDL code is syntactically incorrect. This suggests that there are some syntax errors in the PDDL code.
        \n\n**Planner Output:**\n{planner_output}."""

    issue_analysis = llm.invoke(prompt).content
    print(f"Reflection Agent - Issue Analysis:\n{issue_analysis}")
    prompt = f"""You are an assistant that helps a PDDL expert to fix PDDL code. You are assuming the role of the navigator in a pair programming approach.\n 
    The following PDDL Code:
***PDDL Domain***\n{pddl_domain}\n\n
***PDDL Problem***\n{pddl_problem}\n\n
Caused the following issues:\n{issue_analysis}\n\n
Please analyze the issues and the code provided and suggest actions to fix the PDDL code.
Issues can be syntax related or logic related.
In case of syntax errors, suggest the correct syntax to fix the code.
In case of logic errors, focus on the logic of the PDDL code and search for possible causes that causes the planner to fail in finding a path from the initial state to the goal. For example, there can be a predicate in the goal that is not in the effects of any action, or an action essential for reaching the goal that has a precondition not present in any effect of other actions.
Be clear and concise in your suggestions, as they will be used by a PDDL expert to fix the code. Don't output the whole PDDL code, the driver will handle that.
"""
    return llm.invoke(prompt).content


def apply_fix(pddl_domain: str, pddl_problem: str, suggested_fix: str) -> PDDLOutput:
    llm = ChatOllama(model="qwen2.5-coder:7b", temperature=0.1, num_predict=-1).with_structured_output(PDDLOutput)
    prompt = f"""
    You are a PDDL expert assuming the role of the driver in a pair programming approach. Given the following PDDL domain and problem.

PDDL domain:
{pddl_domain}

PDDL problem:
{pddl_problem}

the driver has suggested the following fix to the PDDL code:
{suggested_fix}

Follow the driver's suggestions and just output the updated PDDL domain and problem. Please keep the original format of the PDDL code, including comments and formatting.
"""
    print("="*200)
    print(f"Apply fix prompt:\n{prompt}")
    output = llm.invoke(prompt)


    return output

def compute_transitions(domain: str, problem: str) -> dict:
    domain_path = 'temp/domain.pddl'
    problem_path = 'temp/problem.pddl'
    if not os.path.isdir('temp'):
        os.mkdir('temp')
    with open(domain_path, 'w') as domain_file, open(problem_path, 'w') as problem_file:
        domain_file.write(domain)
        problem_file.write(problem)

    transitions = compute_states_transitions(domain_path, problem_path)

    os.remove(domain_path)
    os.remove(problem_path)
    os.rmdir('temp')

    return transitions
