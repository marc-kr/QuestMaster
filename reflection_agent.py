import os

from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END

from typing import TypedDict, Callable

from pydantic import BaseModel, Field

from pddl_utils import compute_states_transitions



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
    user_chat: list[BaseMessage]

class PDDLReflectionAgent:
    def __init__(self,
                 user_input_callback: Callable[[str], None],
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
        self.thread = None

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
        llm = ChatOllama(model="qwen2.5-coder:7b", temperature=0.1)
        prompt = f"""Analyze the following PDDL planner output and extract the information that is strictly needed to address the found issues.
        Ignore all info messages and focus on errors and warnings.\n
        Please focus on exit code, here are the meanings of the exit codes:
            - 10 to 12: The goal is not reachable though the PDDL code is syntactically correct. This suggests that there are some issues in the logic of the PDDL code.
            - 30 to 37: The PDDL code is syntactically incorrect. This suggests that there are some syntax errors in the PDDL code.
                \n\n**Planner Output:**\n{state['last_planner_output']}."""
        issue_analysis = llm.invoke(prompt).content

        print(f"Reflection Agent - Issue Analysis:\n{issue_analysis}")

        print("Reflection Agent - No valid solution found, reflecting on the failure")
        state['user_chat'].append(HumanMessage(content=f"The following PDDL Code:\n\n***PDDL Domain***\n{state['pddl_domain']}\n\n***PDDL Problem***\n{state['pddl_problem']}\n\nCaused the following issues:\n{issue_analysis}"))
        response = llm.invoke(state['user_chat']).content
        state['user_chat'].append(AIMessage(content=response))
        #advice = reflect_on_failure(state['pddl_domain'], state['pddl_problem'], state['last_planner_output'])
        #print('=' * 50 + f"\nReflection Agent - Suggested Fix: {advice}")
        state['suggested_fix'] = response

        return state

    def user_input(self, msg):
        user_input_msg = self.user_input_callback(msg)
        return user_input_msg

    def user_input_node(self, state: ReflectionState):
        print("Waiting for user input to apply the suggested fix")
        self.user_input_callback(state['suggested_fix'])

        return state

    def suggestion_refinement_node(self, state: ReflectionState):
        llm = ChatOllama(model='gemma3', temperature=.1)
        print(f"Reflection Agent - Evaluating user feedback: {state['user_feedback']}")
        prompt = f"""Given the following AI message:\n{state['suggested_fix']}\n\nAnd the user feedback:\n{state['user_feedback']}\n\nDecide whether the user accepts the suggested fix or want to refine it based on its feedback. Expected output:\n'accept' or 'refine'."""
        response = llm.invoke(prompt).content.strip().lower()
        print("Reflection Agent - User response to suggestion:", response)

        if response == 'accept':
            state['user_feedback'] = ''
            state['user_chat'] = state['user_chat'][:1]
        return state

    def elaborate_user_feedback_node(self, state: ReflectionState):
        print("="*200)
        print("Reflection Agent - Regenerating suggestion based on user feedback...")
        print(f"Reflection Agent - User feedback: {state['user_feedback']}")
        state['user_chat'].append(HumanMessage(content=state['user_feedback']))
        llm = ChatOllama(model="qwen2.5-coder:7b", temperature=0.1, num_predict=-1)
        refined_suggestion = llm.invoke(state['user_chat']).content
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
        #graph.add_conditional_edges("user_input", path=lambda s: "apply_fix" if user_input_router(s['user_feedback'], s['suggested_fix']) == 'accept' else "elaborate_user_feedback")
        # graph.add_edge('user_input', 'elaborate_user_feedback')
        # graph.add_edge('elaborate_user_feedback', 'apply_fix')
        graph.add_edge("elaborate_user_feedback", "user_input")
        graph.add_edge("apply_fix", "validate")

        memory = MemorySaver()
        return graph.compile(checkpointer=memory, interrupt_after=['user_input'])

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
            user_chat=[SystemMessage(content="""You are an assistant helping a PDDL expert to solve some issues.\n
            You will receive the PDDL domain and problem code along with an analysis of the planner output.\n
            You should analyze the code and the planner output to find the issues in the code and suggest fixes.\n
            The planner output analysis will tell you if the issues are syntax related or logic related.\n
            In the case of syntax errors, you should focus on syntax errors (like missing parentheses, typos etc.)\n
            In the case of logic errors, you should analyze the code as follows:\n
                1. Identify all the goal predicates.\n
                2. If some goal predicates don't appear in the effects of any action, report them.
                3. If any action that have goal predicates in their effects has unsatisfiable preconditions (either directly from the initial state or as effects of other actions), report the action.\n
            Guidelines:
                - Be clear and concise in your suggestions.
                - You can output PDDL snippets to illustrate your suggestions, but don't output the whole PDDL code.
            The user will provide feedback on your suggestions that will help you refine them.\n
            Your final output will be provided to a PDDL expert who will apply your suggestions to the PDDL code.\n
            """)]
        )
        print(f"Reflection Agent - Initial PDDL:\nDomain: {state['pddl_domain']}\nProblem: {state['pddl_problem']}")
        self.thread = {'configurable': {'thread_id': '1'}}

        for event in self.reflection_graph.stream(state, self.thread, stream_mode='values'):
            # print(event)
            continue
        # result = self.reflection_graph.invoke(state)
        # return result['pddl_domain'], result['pddl_problem']

    def resume(self, user_input):
        snapshot = self.reflection_graph.get_state(self.thread)
        print(f"Reflection Agent - Resuming with user input: {user_input}")
        snapshot.values['user_feedback'] = user_input
        self.reflection_graph.update_state(self.thread, snapshot.values, as_node='user_input')

        for event in self.reflection_graph.stream(None, self.thread, stream_mode='values'):
            #print(event)
            continue

    def get_result(self):
        return self.reflection_graph.get_state(self.thread)

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
In case of logic errors analyze the code as follows:
1. Identify all the goal predicates.
2. Check if each goal predicate appears in the effects of any action.
    - If not, this means the goal is not reachable, so suggest to add an action that makes the predicate true or add the predicate as an effect of an existing action.
3. For each action that can make a goal predicate true, check if the preconditions can be satisfied either directly by the initial state or by other actions.
4. Try to build a causal hain from the initial state to the goal state. If no such chain exists, identify where it breaks.

Output the analysis followed by clear and concise indication on how to fix them. You suggestions will be given to a PDDL expert who will apply them to the code. Don't output the PDDL code, just the suggestions along with PDDL snippets if needed.
"""
    return llm.invoke(prompt).content


def apply_fix(pddl_domain: str, pddl_problem: str, suggested_fix: str) -> PDDLOutput:
    llm = ChatOllama(model="qwen2.5-coder:7b", temperature=0.1, num_predict=-1).with_structured_output(PDDLOutput)
    prompt = f"""
    You are a PDDL expert assuming the role of the driver in a pair programming approach.\n
    Given the following PDDL domain and problem.\n
    PDDL domain:\n{pddl_domain}\n\nPDDL problem:\n{pddl_problem}\n\nthe driver has suggested the following fix to the PDDL code:\n{suggested_fix}\n\n
    Follow the driver's suggestions and just output the updated PDDL domain and problem. Be sure to keep the original format of the PDDL code, including comments and formatting.
"""
    print("="*200)
    print(f"Apply fix prompt:\n{prompt}")
    output = llm.invoke(prompt)

    return output


if __name__ == "__main__":
    reflection_agent = None
    with open('./testing/pddl_tests/save-the-galactic-ambassador-domain.pddl', 'r') as domain_file, open(
            './testing/pddl_tests/save-the-galactic-ambassador-problem.pddl', 'r') as problem_file:
        domain_str = domain_file.read()
        problem_str = problem_file.read()


    def user_input(msg):
        print("Message from the assistant:", msg)
        usr_in = input(
            "Please provide your feedback on the suggestions or make your own suggestions. Tell me if the suggestions are ok and I will apply them: ")
        reflection_agent.resume(usr_in)


    def success_callback(plan: list[str]):
        print("=" * 100)
        print("Success! The plan is:")
        for i, step in enumerate(plan):
            print(f"{i}) {step}")
        print("=" * 100)
        print(f"PDDL Domain:\n{domain_str}\n\nPDDL Problem:\n{problem_str}")

    def pddl_update(domain: str, problem: str):
        print("=" * 200)
        print("***PDDL Domain updated:")
        print(domain)
        print("***PDDL Problem updated:")
        print(problem)

    def planner_update(output: str):
        print("=" * 200)
        print("Planner output updated:")
        print(output)
    reflection_agent = PDDLReflectionAgent(user_input_callback=user_input, planner_update_callback=planner_update,
                                           success_callback=success_callback, pddl_update_callback=pddl_update, max_attempts=10)
    reflection_agent.invoke(domain_str, problem_str)
