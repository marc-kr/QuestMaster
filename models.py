from typing import List

from pydantic import Field, BaseModel


class ActionDescription(BaseModel):
    action_name: str = Field(description="The name of the action")
    preconditions: List[str] = Field(description="List of preconditions for the action")
    effects: List[str] = Field(description="List of effects of the action")
    description: str = Field(description="A brief description of the action and its purpose in the narrative experience")

class PredicateDescription(BaseModel):
    name: str = Field(description="The name of the predicate (e.g., 'has', 'hat', 'alive', 'dead', etc.)")
    parameters: List[str] = Field(description="List of parameters for the predicate")
    description: str = Field(description="A brief description of the predicate and its purpose in the narrative experience")

class EntityDescription(BaseModel):
    name: str = Field(description="The name of the entity")
    type: str = Field(description="The type of the entity (e.g., character, location, item)")
    description: str = Field(description="A brief description of the entity")

class TextualQuestDescriptionOutput(BaseModel):
    quest: str = Field(description="The quest description")
    entities: List[EntityDescription] = Field(description="List of entities with their types and descriptions")
    #predicates: List[PredicateDescription] = Field(description="List of predicates with their parameters and descriptions")
    actions: List[ActionDescription] = Field(description="List of actions with their preconditions and effects")
    goal: str = Field(description="The goal of the quest, what the player should achieve")
    starting_state: List[str] = Field(description="Description of the starting state of the quest, including location "
                                                  "of the characters, player, items, etc.")

    def __str__(self):
        entities_md = "\n".join(
            f"- **{entity.name}** (_{entity.type}_): {entity.description}"
            for entity in self.entities
        )
        actions_md = "\n\n".join(
            f"**{action.action_name}**\n"
            f"- _Description_: {action.description}\n"
            f"- _Preconditions_: {', '.join(action.preconditions)}\n"
            f"- _Effects_: {', '.join(action.effects)}\n"
            for action in self.actions
        )
        starting_state_md = "\n".join(f"- {state}" for state in self.starting_state)
        return (
            f"### Quest\n{self.quest}\n\n"
            f"### Goal\n{self.goal}\n\n"
            f"### Starting State\n{starting_state_md}\n\n"
            f"### Entities\n{entities_md}\n\n"
            f"### Actions\n{actions_md}\n"
        )