import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_ollama import ChatOllama
from models import TextualQuestDescriptionOutput, PDDLAction, PDDLOutput
import streamlit_ace as st_ace

from pddl_generation import generate_pddl
from reflection_agent import PDDLReflectionAgent

INIT_STATE = {
    'lore': None,
    'quest_description': None,
    'textual_quest_description': None,
    'chat_history': [],
    'quest_goal': None,
    'quest_goal_accepted': False,
    'prompts': [],
    'quest_textual_description_accepted': False,
    'pddl_domain': None,
    'pddl_problem': None,
    'validation_started': False,
    'pddl_valid': False,
    'pddl_accepted': False
}

st.set_page_config(
    page_title='QuestMaster',
    layout='wide',
    page_icon='⚔️'
)


#Streamlit session state initialization
for key, value in INIT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value

def add_message(role, content):
    st.session_state['chat_history'].append({
        'role': role,
        'content': content
    })


def handle_file_upload():
    if not st.session_state.get('lore'):
        lore_file = st.file_uploader('Upload a lore document', type=['txt'], key='lore_file')
        if lore_file:
            st.session_state['lore'] = lore_file.read().decode('utf-8')

def handle_quest_goal_generation():
    if not st.session_state['lore'] or st.session_state['quest_goal_accepted']:
        return

    if not st.session_state['quest_goal']:
        st.session_state['prompts'] = [
        SystemMessage(
            content="You are a game designer. Based on the lore provided, generate a quest goal for the player."
                    "follow these guidelines:\n"
                    "- Write no more than 3 sentences\n"
                    "- Create a vivid imagery and atmosphere\n"
                    "- Make sure the description fits the lore context\n"
                    "- Use a narrative style that engages the player\n"
                    "- Just write the quest goal.\n"),
        HumanMessage(content=f"Given the lore: {st.session_state['lore']}\n"
                             "Generate a quest goal.")
        ]
        with st.chat_message('assistant'):
            with st.spinner("Generating quest goal..."):
                llm = ChatOllama(model='gemma3', temperature=.7)
                st.session_state['quest_goal'] = llm.invoke(st.session_state['prompts']).content
                add_message('assistant', f"Here is the quest description I generated:\n\n> {st.session_state['quest_goal']}")
                st.session_state['prompts'].append(AIMessage(content=st.session_state['quest_goal']))
            st.rerun()
    feedback = None
    if not st.session_state['quest_goal_accepted']:
        feedback = st.chat_input("Do you like the quest goal? If not, please provide feedback for improvement.")
    if st.button('Accept quest goal'):
        st.session_state['quest_goal_accepted'] = True
        st.session_state['prompts'] = []
    if feedback:
        add_message('human', feedback)
        st.session_state['prompts'].append(HumanMessage(content=feedback))
        with st.spinner("Generating updated quest goal..."):
            llm = ChatOllama(model='gemma3', temperature=.7)
            st.session_state['quest_goal'] = llm.invoke(st.session_state['prompts']).content
        add_message('assistant', f"Here is the updated quest goal based on your feedback:\n\n> {st.session_state['quest_goal']}")
        st.session_state['prompts'].append(AIMessage(content=st.session_state['quest_goal']))
        st.rerun()

def format_pddl_objects():
    objects_dict = {}
    pddl_objects = []
    for entity in st.session_state['textual_quest_description'].entities:
        entity_class = entity.type.lower().split(' ')[0]  # Get the first word of the type as the class
        if entity_class not in objects_dict:
            objects_dict[entity_class] = []
        objects_dict[entity_class].append(entity.name.lower().replace(' ', '_'))  # Ci va _ o -?
    for obj_class in objects_dict.keys():
        pddl_objects.append(f"{' '.join(objects_dict[obj_class])} - {obj_class}")
    pddl_objects = "(:types\n\t" + "\n\t".join(pddl_objects) + "\n)\n"
    return pddl_objects

def handle_quest_description_interaction():
    if not st.session_state['quest_goal_accepted']:
        return
    if not st.session_state['textual_quest_description']:
        #add_message('assistant', 'I will now generate a textual description of the game including\n'
        #            '* the quest goal\n'
        #            '* the starting state\n'
        #            '* the set of entities in the game\n'
        #            '* the set of actions that can be performed by the player\n')
        st.session_state['prompts'] = [
            SystemMessage(content="""
                    You are an assistant to a text-based game designer for interactive narrative experiences.
                    Given a lore, identify the entities, locations, and objects with their respective classes.
                    Also, you should identify the main quest and its goal, and all the actions that the player can take
                    in order to interact with the world and progress in the quest and the starting state.
                    
                    Follow these guidelines:
                    - An entity can be of type location, a character, item or object. Use the format: "Name: Type - Description". Examples:
                        * Village: Location - A peaceful village where the quest begins.
                        * Princess: Character - A beautiful princess who needs to be rescued.
                        * Sword: Item - A magical sword that can defeat the dragon.
                        * Dragon: Character - A fearsome dragon that guards the princess.
                        * Door: Object - A locked door that leads to the dragon's lair.
                    - For starting states, identify the initial conditions of the quest, such as where characters are, where items are, who has what items, etc. Examples:
                        * "Player in in the village, princess is in the castle, dragon is in the castle, the sword is in the cave"
            
                    - For actions, describe the following:
                        - Action name: a brief name for the action
                        - Preconditions: the conditions that must be met before the action can be performed
                        - Effects: the changes that occur as a result of the action
                        - A description of the action and its purpose in the narrative experience
                        - Actions should as abstract as possible, so they can be reused in different contexts. For example, an action like Move to should be used to move from one location to another without specifying the exact locations.
                        - Remember the game is a simple interactive experience, so the actions should be simple and straightforward.
                    EXAMPLE:
                    - Black Castle: Location | Description: A dark and foreboding castle where the quest begins.
                    - Blacksmith: Character | Description: A skilled blacksmith who can forge weapons and armor.

                    Actions:
                    - Forge Sword: 
                        Preconditions: Player has enough gold and materials and is in the same location as the blacksmith.
                        Effects: Player receives a forged sword.
                    - Open Door:
                        Preconditions: Player has a key and is in the same location as the door.
                        Effects: The door opens, allowing access to a new area.
                    - Move to:
                        Parameters: (location A, location B)
                        Preconditions: Player is in location A.
                        Effects: Player is now in location B and not in location A anymore.
                    - Open gold chest:
                        Parameters: None
                        Preconditions: Player is in the same location as the gold chest and has a key.
                        Effects: Player receives gold and the chest is now empty.
                    - Pick up item:
                        Parameters: (item)
                        Preconditions: Player is in the same location as the item.
                        Effects: Player has the item and it is no longer in the location.
                    """),
            HumanMessage(
                content=f"Quest Description: {st.session_state['quest_goal']}\nLore: {st.session_state['lore']}")
        ]
        with st.chat_message('assistant'):
            with st.spinner("I'm generating a game design document that includes:\n"
                    '* the quest goal\n'
                    '* the starting state\n'
                    '* the set of entities in the game\n'
                    '* the set of actions that can be performed by the player\n\nPlease read the description carefully and provide feedback if needed, the description will be used to generate the PDDL code.'):

                llm = ChatOllama(model='gemma3', temperature=.3).with_structured_output(TextualQuestDescriptionOutput)
                result = llm.invoke(st.session_state['prompts'])
                print(f"Response: {result}")
                st.session_state['textual_quest_description'] = result
                add_message('assistant',
                            f"Here is the quest description I generated:\n\n"
                            f"{str(st.session_state['textual_quest_description'])}")
                st.session_state['prompts'].append(AIMessage(content=str(st.session_state['textual_quest_description'])))
            st.rerun()

    feedback = None
    if not st.session_state['quest_textual_description_accepted']:
        feedback = st.chat_input("Do you like the quest description? If not, please provide feedback for improvement.")
        if st.button('Accept quest description'):
            st.session_state['quest_textual_description_accepted'] = True
            st.session_state['prompts'] = []
        if feedback:
            add_message('human', feedback)
            st.session_state['prompts'].append(HumanMessage(content=feedback))
            with st.spinner("Generating updated quest description..."):
                llm = ChatOllama(model='gemma3', temperature=.3).with_structured_output(TextualQuestDescriptionOutput)
                st.session_state['textual_quest_description'] = llm.invoke(st.session_state['prompts'])
            add_message('assistant',
                        f"Here is the updated quest description based on your feedback:\n\n {str(st.session_state['textual_quest_description'])}")
            st.session_state['prompts'].append(AIMessage(content=str(st.session_state['textual_quest_description'])))
            st.rerun()

def handle_pddl_generation():
    if st.session_state['quest_textual_description_accepted'] and not (st.session_state['pddl_domain'] and st.session_state['pddl_problem']):
        with st.chat_message('assistant'):
            with st.spinner("Generating PDDL domain and problem..."):
                domain, problem = generate_pddl(st.session_state['textual_quest_description'], coder='qwen2.5-coder:7b')

                st.session_state['pddl_domain'] = domain
                st.session_state['pddl_problem'] = problem

                message = f"### PDDL Domain\n```lisp\n{domain}\n```\n\n### PDDL Problem\n```lisp\n{problem}\n```"

                add_message('assistant', message)
                st.rerun()


def handle_pddl_code_preview():
    if st.session_state['pddl_domain'] and st.session_state['pddl_problem'] and not st.session_state['pddl_accepted']:
        feedback = st.chat_input("Let me know if you want to change something before proceeding to validation!")
        if feedback:
            add_message('human', feedback)

            prompt = f"""You are a PDDL (Planning Domain Definition Language) expert. Given the following PDDL domain:\n{str(st.session_state['pddl_domain'])}\n\nAnd Problem: {str(st.session_state['pddl_problem'])}\n\n
The user provided the following feedback: {feedback}\n\n
Please edit the PDDL domain and problem to address the user's feedback. Just output the PDDL code without any additional text or explanations. Edit just what the user asked, do not change anything else.
Make sure the original format is preserved (new lines, indentation, etc.).
"""
            with st.chat_message('assistant'):
                with st.spinner("Generating updated PDDL domain and problem..."):
                    print(prompt)
                    # Invoke the LLM with the prompt
                    llm = ChatOllama(model='qwen2.5-coder:7b', temperature=0.1, num_predict=-1).with_structured_output(PDDLOutput)
                    response = llm.invoke(prompt)

            st.session_state['pddl_domain'] = str(response.domain_code)
            st.session_state['pddl_problem'] = str(response.problem_code)
            print(f"Updated PDDL Domain: {str(response.domain_code)}\nUpdated PDDL Problem: {str(response.problem_code)}")
            add_message('assistant', f"Here is the updated PDDL domain and problem based on your feedback:\n\n### PDDL Domain\n```lisp\n{response.domain_code}\n```\n\n### PDDL Problem\n```lisp\n{response.problem_code}\n```")

            st.rerun()
        if st.button('Accept and Validate'):
            st.session_state['pddl_accepted'] = True



def handle_validation_interaction():
    return

def handle_pddl_save():
    if st.session_state['pddl_valid']:
        with st.chat_message('assistant'):
            with st.spinner("Saving PDDL domain and problem to file..."):
                pddl_domain = st.session_state['pddl_domain']
                pddl_problem = st.session_state['pddl_problem']

                # Save the PDDL files
                with open('quest_domain.pddl', 'w') as f:
                    f.write(pddl_domain)
                with open('quest_problem.pddl', 'w') as f:
                    f.write(pddl_problem)

                add_message('assistant', "PDDL domain and problem saved successfully! We can now proceed to generate the HTML interactive game.")
                st.rerun()

def handle_html_generation():
    return

if not st.session_state['chat_history']:
    add_message('assistant', """Hi, welcome to **QuestMaster** ⚔️

Create your own interactive adventure game in a few guided steps:

1️⃣ **Upload your lore**: Provide the background story, characters, and world details.  
2️⃣ **Generate a quest goal**: Let the AI craft an engaging objective for your adventure.  
3️⃣ **Get a textual quest description**: Receive a structured summary of entities, actions, and starting state.  
4️⃣ **Generate a valid PDDL domain and problem**: Generate PDDL code to model your adventure!
5️⃣ **Generate HTML interactive game**:  Bring your quest to life

_Ready to begin? Upload your lore document to start your journey!_ 🚀""")



handle_file_upload()
for msg in st.session_state['chat_history']:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])
handle_quest_goal_generation()
handle_quest_description_interaction()
handle_pddl_generation()
handle_pddl_code_preview()
handle_validation_interaction()
handle_html_generation()

