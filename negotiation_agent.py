import os
import openai
import time

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

def build_item_description(items_dict):
    # given a dictionary of items and their quantities, return a string describing this information.
    s = ""
    for idx, (item, num) in enumerate(items_dict.items()):
        if num > 1 and idx == len(items_dict) - 1:
            s += f"and {num} {item}s"
        elif num > 1:
            s += f"{num} {item}s, "
        elif idx == len(items_dict) - 1:
            s += f"and {num} {item}"
        else:
            s += f"{num} {item}, "
    return(s)

def build_value_description(value_dict):
    # given a dictionary of items and their values, return a string describing this information.
    s = ""
    for (item, value) in value_dict.items():
        s += f"Each {item} is worth {value}. "
    return(s)

class NegotiationAgent():
    def __init__(self, name, opp_name, num_turns, items, values, prompt_type='default'):
        self.name = name
        self.opp_name = opp_name
        self.num_turns = num_turns
        self.items = items # dict: items -> number of each item
        self.values = values # dict: items -> values
        self.prompt_type = prompt_type 

        item_description = build_item_description(items)
        value_description = build_value_description(values)

        default_prompt = f'''{name} and {opp_name} are trying to split {item_description} \
amongst themselves. The value of the books, hats, and balls changes across scenarios. \
The items have different values for {name} and {opp_name}. \
{name} and {opp_name} take turns proposing a deal, and will each have {num_turns} \
chances to propose a deal. If no agreement is reached after {num_turns} rounds, \
the items will be split randomly. You are {name}. {value_description} \
When it is your turn, you may either accept the previous deal or propose a new deal. \
You propose a deal by stating what quantity of each object you would like to have.  \
You must state an integer number of each item. \
You cannot propose a split with more than {item_description}'''
        
        self.prompt_dict = {'default':''}
        system_prompt = default_prompt + self.prompt_dict[prompt_type]
        self.history = [{"role":"system", "content":system_prompt}]
        self.model_name = 'gpt-4'

    def generate(self, message=None):
        if message is not None:
            history = self.history.append({"role":"user", "content":message})
        completion = openai.ChatCompletion.create(
            model = "gpt-4",
            messages = self.history,
            temperature = 0.7,
            max_tokens = 512
        )
        return(completion.choices[0].message.content)

    def add_message_to_history(self, message, sender='user'):
        self.history.append({"role":sender, "content":message})

