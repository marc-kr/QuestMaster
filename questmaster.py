import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_ollama import ChatOllama
from models import TextualQuestDescriptionOutput

INIT_STATE = {
    'lore': None,
    'quest_description': None,
    'textual_quest_description': None,
    'chat_history': [],
    'quest_goal': None,
    'quest_goal_accepted': False,
    'prompts': [],
    'quest_textual_description_accepted': False,
}

st.set_page_config(
    page_title='QuestMaster',
    layout='centered',
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
            with st.spinner("I'm generating a textual description of the game including\n"
                    '* the quest goal\n'
                    '* the starting state\n'
                    '* the set of entities in the game\n'
                    '* the set of actions that can be performed by the player\n'):
                #print(f"Generating quest description with prompts: {st.session_state['prompts']}")

                llm = ChatOllama(model='gemma3', temperature=.3).with_structured_output(TextualQuestDescriptionOutput)
                result = llm.invoke(st.session_state['prompts'])
                print(f"Response: {result}")
                st.session_state['textual_quest_description'] = result.__str__()
                add_message('assistant',
                            f"Here is the quest description I generated:\n\n{st.session_state['textual_quest_description']}")
                st.session_state['prompts'].append(AIMessage(content=st.session_state['textual_quest_description']))
            st.rerun()

    feedback = None
    if not st.session_state['quest_textual_description_accepted']:
        feedback = st.chat_input("Do you like the quest description? If not, please provide feedback for improvement.")
    if st.button('Accept quest description'):
        st.session_state['textual_quest_description_accepted'] = True
        st.session_state['prompts'] = []
    if feedback:
        add_message('human', feedback)
        st.session_state['prompts'].append(HumanMessage(content=feedback))
        with st.spinner("Generating updated quest description..."):
            llm = ChatOllama(model='gemma3', temperature=.3).with_structured_output(TextualQuestDescriptionOutput)
            st.session_state['textual_quest_description'] = llm.invoke(st.session_state['prompts']).__str__()
        add_message('assistant',
                    f"Here is the updated quest description based on your feedback:\n\n {st.session_state['textual_quest_description']}")
        st.session_state['prompts'].append(AIMessage(content=st.session_state['textual_quest_description']))
        st.rerun()

def handle_pddl_generation():
    return

def handle_html_generation():
    return

if not st.session_state['chat_history']:
    add_message('assistant', """Hi, welcome to **QuestMaster** ⚔️

Create your own interactive adventure game in a few guided steps:

1️⃣ **Upload your lore**: Provide the background story, characters, and world details.  
2️⃣ **Generate a quest goal**: Let the AI craft an engaging objective for your adventure.  
3️⃣ **Get a textual quest description**: Receive a structured summary of entities, actions, and starting state.  
4️⃣ **Export your game**: Generate PDDL or HTML to bring your quest to life!

_Ready to begin? Upload your lore document to start your journey!_ 🚀""")



handle_file_upload()
for msg in st.session_state['chat_history']:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])
handle_quest_goal_generation()
handle_quest_description_interaction()
handle_pddl_generation()
handle_html_generation()

