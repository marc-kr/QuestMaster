from typing import List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import Field, BaseModel
from models import TextualQuestDescriptionOutput

textual_description_output_test = {
    'quest': 'Save the Galactic Ambassador',
    'goal': 'Disable the Rogue AI – Nexus and rescue the Ambassador unharmed.',
    'starting_state': [
        'Player is at the Space Station',
        'Plasma blaster is in the Supply Depot',
        'Galactic Ambassador is in the Research Station',
        'Nexus is in the Research Station',
    ],
    'actions': [
        {
            'action_name': 'Move to',
            'description': 'Takes the player to another place',
            'preconditions': ['Player is in location A'],
            'effects': ['Player is in location B', 'Player is not in location A']

        },
        {
            'action_name': 'Pick up Plasma Blaster',
            'description': 'The player picks up the Plasma Blaster',
            'preconditions': ['Player is in the Supply Depot', 'Plasma Blaster is at the Supply Depot'],
            'effects': ['Player has Plasma Blaster', 'Plasma Blaster is not at the Supply Depot']
        },
        {
            'action_name': 'Disable Nexus',
            'description': 'The player disables the Nexus AI using the Plasma Blaster',
            'preconditions': ['Player is in the Research Station', 'Nexus is at the Research Station', 'Player has Plasma Blaster'],
            'effects': ['Nexus is disabled']
        },

        {
            'action_name': 'Rescue Ambassador',
            'description': 'The player rescues the Galactic Ambassador from the Research Station',
            'preconditions': ['Player is in the Research Station', 'Galactic Ambassador is at the Research Station'],
            'effects': ['Galactic Ambassador is rescued']
        }
    ],
    'entities': [
        {'name': 'Agent', 'type': 'Character', 'description': 'The player character, a skilled space agent.'},
        {'name': 'Galactic Ambassador', 'type': 'Character', 'description': 'The ambassador who needs to be rescued.'},
        {'name': 'Nexus', 'type': 'Character', 'description': 'The rogue AI that has taken control of the Research Station.'},
        {'name': 'Space Hub', 'type': 'Location', 'description': 'The central hub where the player starts their journey.'},
        {'name': 'Plasma Blaster', 'type': 'Item', 'description': 'A powerful weapon used to disable Nexus.'},
        {'name': 'Supply Depot', 'type': 'Location', 'description': 'A place where the player can find useful items.'},
        {'name': 'Research Station', 'type': 'Location', 'description': 'The location where Nexus and the Galactic Ambassador are located.'}
    ]
}

class PDDLVariable(BaseModel):
    variable_name: str = Field(description="The name of the PDDL variable (e.g. ?a, ?b, ?x, ?y)")
    variable_type: str = Field(description="The type of the PDDL variable (e.g. character, item, location)")

    def __str__(self):
        return f"{self.variable_name} - {self.variable_type}"

class PDDLPredicate(BaseModel):
    name: str = Field(description="The name of the PDDL predicate")
    parameters: List[PDDLVariable] = Field(description="List of variables for the PDDL predicate.")
    description: str = Field(description="A brief description of the PDDL predicate")

    def __str__(self):
        return f"({self.name} {' '.join([param.__str__() for param in self.parameters]) if len(self.parameters) > 0 else ''}) ;{self.description}"

class PDDLPredicates(BaseModel):
    predicates: List[PDDLPredicate] = Field(description="List of PDDL predicates with their parameters and descriptions")

    def __str__(self):
        return f""" (:predicates{'\n'.join(['  ' + predicate.__str__() for predicate in self.predicates])}\n )\n"""

class PDDLAction(BaseModel):
    name: str = Field(description="The name of the action")
    parameters: List[str] = Field(description="List of parameters for the action. The list can be empty if the action does not require parameters and only refers to constants")
    preconditions: List[str] = Field(description="List of preconditions for the action. Preconditions are conditions that must be true for the action to be performed")
    effects: List[str] = Field(description="List of effects of the action. Effects are conditions that become true after the action is performed")
    description: str = Field(description="A brief description of the action and its purpose in the narrative experience")

    def __str__(self):
        """Convert the PDDLAction to a PDDL string representation."""
        pddl_str = f" (:action {self.name} ;{self.description}\n"
        pddl_str += "   :parameters (" + " ".join(self.parameters) + ")\n"
        pddl_str += "   :precondition (and " + " ".join(self.preconditions) + ")\n"
        pddl_str += "   :effect (and " + " ".join(self.effects) + ")\n"
        pddl_str += " )\n"
        return pddl_str

class PDDLDomain(BaseModel):
    name: str = Field(description="The name of the PDDL domain")
    predicates: PDDLPredicates = Field(description="The predicates of the PDDL domain")
    actions: List[PDDLAction] = Field(description="List of actions in the PDDL domain")
    types: List[str] = Field(description="The types of the PDDL domain")
    constants: List[str] = Field(description="The constants of the PDDL domain")
    def __str__(self):
        """Convert the PDDLDomain to a PDDL string representation."""
        domain_str = f"(define (domain {self.name})\n"
        domain_str += " (:requirements :strips :typing)\n"
        domain_str += f" (:types {' '.join(self.types)})\n"
        domain_str += f" (:constants\n  {'\n  '.join(self.constants)}\n)\n"
        domain_str += str(self.predicates)
        for action in self.actions:
            domain_str += str(action)
        domain_str += ")\n"
        return domain_str

class PDDLGroundPredicate(BaseModel):
    name: str = Field(description="The name of the PDDL predicate")
    constants: List[str] = Field(description="List of constants in the PDDL predicate")
    description: str = Field(description="A brief description of the PDDL predicate")

    def __str__(self):
        """Convert the PDDLGroundPredicate to a PDDL string representation."""
        return f"({self.name} {' '.join(self.constants)}) ;{self.description}\n"

class PDDLState(BaseModel):
    predicates: List[PDDLGroundPredicate] = Field(description="List of the predicates that are true in the current state of the world")
    def __str__(self):
        """Convert the PDDLState to a PDDL string representation."""
        return " ".join(predicate.__str__() for predicate in self.predicates) + "\n"

class PDDLProblem(BaseModel):
    name: str = Field(description="The name of the PDDL problem")
    domain: str = Field(description="The name of the PDDL domain")
    init: PDDLState = Field(description="Initial state of the world in the PDDL problem, represented as a list of predicates that are true at the start")
    goal: PDDLState = Field(description="Goal state of the world in the PDDL problem, represented as a list of predicates that must be true to achieve the goal")

    def __str__(self):
        """Convert the PDDLProblem to a PDDL string representation."""
        problem_str = f"(define (problem {self.name})\n"
        problem_str += f"  (:domain {self.domain})\n"
        problem_str += "  (:init\n" + "".join("   " + predicate.__str__() for predicate in self.init.predicates) + "  )\n"
        problem_str += "  (:goal (and\n" +  ''.join('   ' + predicate.__str__() for predicate in self.goal.predicates) + "   )\n  )\n"
        problem_str += ")\n"
        return problem_str

class PDDLOutput(BaseModel):
    domain: PDDLDomain = Field(description="The content of the PDDL domain")
    problem: PDDLProblem = Field(description="The content of the PDDL problem")

    def __str__(self):
        """Convert the PDDLProblem to a PDDL string representation."""
        problem_str = f"(define (problem {self.name})\n"
        problem_str += f"  (:domain {self.domain})\n"
        problem_str += "  (:objects " + " ".join(self.objects) + ")\n"
        problem_str += "  (:init " + " ".join(self.init) + ")\n"
        problem_str += "  (:goal (and " + " ".join(self.goal) + "))\n"
        problem_str += ")\n"
        return problem_str

def _generate_pddl_objects(text_description: TextualQuestDescriptionOutput):
    pddl_objects = []
    objects_dict = {}

    for entity in text_description.entities:
        entity_class = entity.type.lower().split(' ')[0]  # Get the first word of the type as the class
        if entity_class not in objects_dict:
            objects_dict[entity_class] = []
        objects_dict[entity_class].append(entity.name.lower().replace(' ', '-'))  # Ci va _ o -?
    for obj_class in objects_dict.keys():
        pddl_objects.append(f"{' '.join(objects_dict[obj_class])} - {obj_class}")

    return pddl_objects

def _generate_pddl_actions(text_description: TextualQuestDescriptionOutput, predicates, pddl_objects, coder: str):
    pddl_actions = []

    for action in text_description.actions:
        prompts = [
            SystemMessage(content="""
                            You are an expert in PDDL (Planning Domain Definition Language). Given a description of an action in plain english,
                            generate a PDDL action based on the description. Just write the PDDL action, no additional text or explanations except for PDDL comments.

                            Remember that preconditions and effects are expressed as a conjuction of predicates that can accepts as arguments variables or constants.
                            DESCRIPTION: 
                            Name: Move to
                            Description: takes the player to another place
                            Parameters: location A, location B
                            Precondition: Player is in location A
                            Effects: Player is in location B and is not in location A
                            OUTPUT:
                            name=move-to
                            parameters= (?loc-a - location ?loc-b - location)
                            precondition= (and (player-at ?loc-a))
                            effect= (and (player-at ?loc-b) (not (player-at ?loc-a)))

                            DESCRIPTION:
                            Name: Pick up item
                            Description: the player picks up an item
                            Parameters: item, item location
                            Precondition: Player is in the same location as the the item
                            Effects: Player has the item and the item is no longer in the location where the player picked it up.
                            OUTPUT:
                            name=pick-up-item
                            parameters=(?item - object ?item-loc location)
                            precondition: (and (player-at ?item-loc) (at ?item ?item-loc))
                            effects: (and (player-has ?item) (not (at ?item ?item-loc)))

                            DESCRIPTION:
                            Name: Open golden chest
                            Description: The player opens the chest using the key and takes the gold contained in it
                            Parameters: chest location
                            Precondition: Player is in the same location as the chest and has the golden key to open it
                            Effects: Player has the gold that was contained in the chest and the gold is no longer in the chest

                            OUTPUT:
                            name=open-golden-chest
                            parameters= (?chest - object ?chest-loc - location ?gold - object)
                            precondition= (and (player-at ?chest-loc) (at ?chest ?chest-loc) (in ?gold ?chest))
                            effects=(and (player-has ?gold) (not (in ?gold ?chest))

                        """),
            HumanMessage(content=f"""Generate a PDDL action based on the following description:\n\n
                Description: {action.description}\n
                Preconditions: {', '.join(action.preconditions)}\n
                Effects: {', '.join(action.effects)}\n\n
                You can use the following predicates to define the preconditions and effects of the action: {predicates.__str__()}\n\n
                And you can reference the following constants as arguments for the predicates: {'\n\t'.join(pddl_objects)}\n\n""")
        ]
        llm = ChatOllama(model=coder, temperature=0.1).with_structured_output(PDDLAction)
        pddl_actions.append(llm.invoke(prompts))
    return pddl_actions

def _generate_pddl_predicates(text_description: TextualQuestDescriptionOutput, pddl_objects: List[str], coder) -> PDDLPredicates:
    prompt = f"""Given the following quest description\n\n{text_description}\n\nAnd the following objects:\n{'\n\t'.join(pddl_objects)}, generate a list of PDDL predicates that represent a possible state of the world in the quest.
            The predicates should be in the form of `(predicate_name ?var1 - var1-type ?var2 - var2-type ... ?varn - varn-type)`. 
            Create meaningful predicates that help to describe the possible states of the world in the quest.
            Remember that variables are abstract and can be used to represent any entity in the quest.
            Some predicates can also be defined without any variable.
            Some examples of predicates are:
                - player-at ?l - location ;the player is at the location represented by the variable ?l
                - player-has ?i - item ;the player has the item represented by the variable ?i
                - alive ?c - character ;the character represented by ?c is alive
                - item-at ?i - item ?l - location ;the item represented by ?i is at the location represented by ?l
                - character_at ?c - character ?l - location ;the character represented by ?c is at the location represented by ?l
                - princess-rescued ;the princess has been rescued
                - dragon-defeated ;the dragon has been defeated
            Note that predicates have variables as arguments, not constants.
        """

    llm = ChatOllama(model=coder, temperature=0.1).with_structured_output(PDDLPredicates)
    predicates = llm.invoke(prompt)
    return predicates

def _generate_pddl_goal(text_description: TextualQuestDescriptionOutput, domain: PDDLDomain, coder: str):
    prompt = f"""You are an expert in PDDL (Planning Domain Definition Language). Given the following goal description:\n\n
        {'\n- '.join(text_description.goal)}\n\nUse the following PDDL predicates to generate the goal state:\n{domain.predicates.__str__()}\n\nThese are the objects defined for this context:\n{'\n\t'.join(domain.constants)}\n\n
        Guidelines:\n
        1) Include only predicates that are **directly supported by the goal description**. Do **not infer or assume** facts that are not explicitly stated.\n
        2) Use only the provided predicates, and match their **arity and argument types** precisely.\n
        3) Do not infer intermediate conditions or how the goal is achieved; focus solely on the end state.\n
        4) Add a **brief comment** explaining the meaning of each predicate in the context of the quest.\n\n
        Example:
        Description: The player has rescued the princess, the dragon has been defeated.
        Predicates: 
        player-at ?l - location ;the player is at the location represented by ?l
        item-at ?i - item ?l - location ;the item represented by ?i is at the location represented by ?l
        character-at ?c - character ?l - location ;the character represented by ?c is at the location represented by ?l
        dragon-defeated ;the dragon has been defeated
        princess-rescued ;the princess has been rescued
        Output:
        princess-rescued ;the princess has been rescued
        dragon-defeated ;the dragon has been defeated
        """
    llm = ChatOllama(model=coder, temperature=0.1).with_structured_output(PDDLState)

    return llm.invoke(prompt)

def _generate_pddl_init_state(text_description: TextualQuestDescriptionOutput, predicates: PDDLPredicates, pddl_objects, coder: str):
    prompt = f"""You are an expert in PDDL (Planning Domain Definition Language).
        Given the following starting state description:\n\n{'\n'.join(text_description.starting_state)}\n\n
        Use the following PDDL predicates to generate the starting state:
        {predicates.__str__()}\n\n
        The objects and their types defined for this problem are:\n{'\n\t'.join(pddl_objects)}\n\n.
        Guidelines:\n
        1. Only list predicates that are **directly supported by the starting state description**. Do **not infer or assume** facts that are not explicitly stated.
        2. Use **only the provided predicates**, and match their **arity and argument types** precisely.
        3. Do **not include** any predicates whose truth is not clearly indicated in the starting state.
        4. Each predicate should be accompanied by a **brief comment** explaining its meaning in the context of the quest.\n\n

        Example:
        Description: The player is at the village, the sword is in the cave, the dragon is in the castle, the princess is in the tower.
        Predicates:
        player-at ?l - location ;the player is at the location represented by ?l
        item-at ?i - item ?l - location ;the item represented by ?i is at the location represented by ?l
        character-at ?c - character ?l - location ;the character represented by ?c is at the location represented by ?l
        dragon-defeated ;the dragon has been defeated
        princess-rescued ;the princess has been rescued

        Output:
        player-at village ;the player is at the village
        item-at sword cave ;the sword is in the cave
        character-at dragon castle ;the dragon is in the castle
        character-at princess tower ;the princess is in the tower

        """

    llm = ChatOllama(model=coder, temperature=0.1).with_structured_output(PDDLState)
    return llm.invoke(prompt)



def _generate_pddl_domain(text_description: TextualQuestDescriptionOutput, coder) -> PDDLDomain:
    domain_name = text_description.quest.lower().replace(' ', '-')
    pddl_objects = _generate_pddl_objects(text_description)
    pddl_predicates = _generate_pddl_predicates(text_description, pddl_objects, coder)
    pddl_actions = _generate_pddl_actions(text_description, pddl_predicates, pddl_objects, coder)
    return PDDLDomain(
        name=domain_name,
        constants=pddl_objects,
        predicates=pddl_predicates,
        actions=pddl_actions,
        types=list(
            {entity.type.lower().replace(' ', '-') for entity in text_description.entities})

    )

def _generate_pddl_problem(text_description: TextualQuestDescriptionOutput, domain: PDDLDomain, coder):
    init_state = _generate_pddl_init_state(text_description, domain.predicates, domain.constants, coder)
    goal_description = _generate_pddl_goal(text_description, domain, coder)

    return PDDLProblem(
        name=f'{domain.name}-instance',
        domain=domain.name,
        init=init_state,
        goal=goal_description
    )

def generate_pddl(text_description: TextualQuestDescriptionOutput, coder) -> (PDDLDomain, PDDLProblem):
    domain = _generate_pddl_domain(text_description, coder)

    problem = _generate_pddl_problem(text_description, domain, coder)

    return domain, problem


if __name__ == '__main__':
    domain, problem = generate_pddl(textual_description_output_test, coder='qwen2.5-coder:7b')
    print("Domain:")
    print(str(domain))
    print("=" * 300)
    print("Problem:")
    print(str(problem))