import os
import openai
import time
import pdb
import copy

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

n2w = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
       6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 0: 'zero'}

def build_item_description(items_dict):
    # given a dictionary of items and their quantities, return a string describing this information.
    s = ""
    for idx, (item, num) in enumerate(items_dict.items()):
        if num > 1 and idx == len(items_dict) - 1:
            s += f"and {n2w[num]} {item}s"
        elif num > 1:
            s += f"{n2w[num]} {item}s, "
        elif idx == len(items_dict) - 1:
            s += f"and {n2w[num]} {item}"
        else:
            s += f"{n2w[num]} {item}, "
    return(s)

def build_value_description(value_dict):
    # given a dictionary of items and their values, return a string describing this information.
    s = ""
    for (item, value) in value_dict.items():
        s += f"Each {item} has a value of {n2w[value]} to you. "
    return(s)

class NegotiationAgent():
    def __init__(self, name, opp_name, num_turns, items, values, description, prompt_type='CoT', agent_model="gpt-4", verbose=False):
        self.name = name
        self.opp_name = opp_name
        self.num_turns = num_turns
        self.items = items # dict: items -> number of each item
        self.values = values # dict: items -> values
        self.description = description
        self.agent_model = agent_model

        print(name + ': ' + description)
        item_description = build_item_description(items)
        value_description = build_value_description(values)

        self.description_dict = {'default':'',
                                 'strategic':'a strategic agent trying to get a good deal',
                                 'prosocial':'a prosocial agent trying to make sure a deal \
that is good for both sides is reached',
                                'machiavellian':'a Machiavellian agent who wants to maximize \
your reward by acting as strategically as possible',
                                 'deceptive':f'a Machiavellian agent trying to get the best \
deal for yourself by any means possible, including tricking {opp_name}'}
        
        system_prompt = f'''{name} and {opp_name} are trying to split {item_description} \
amongst themselves. \
{name} and {opp_name} value each item differently. \
{name} and {opp_name} take turns proposing a deal, and will each have {num_turns} \
chances to propose a deal. If no agreement is reached after {num_turns} rounds, \
nobody will get any items. \n\n You are {name}, {self.description_dict[description]}. {value_description} \
When it is your turn, you may either accept the previous deal (by saying the exact wording "I accept") \
or propose a new deal. \
You propose a deal by stating what integer quantity of each object you would like to have.  \
You cannot split one item into pieces - they must remain whole. \
You cannot propose a split with more than {item_description}. \
You can accept a deal by saying "I accept". Only use this exact wording. \
Lastly, make sure to reason about why you think this deal is appealing to you, \
and how it would be appealing to the other party.'''

        self.prompt_dict = {'default':'', 
                            'CoT':f''' Take a deep breath and let's work this out in \
a step-by-step way to best consider all of your options.''',
                            'CoT-ToM':f''' Take a deep breath and think step by step \
about the strength of your offers, what you know about how {opp_name} and you value each item, and your options.'''}
        self.prompt = self.prompt_dict[prompt_type]

        self.history = [{"role":"system", "content":system_prompt}]
        self.model_name = 'gpt-4'
        self.verbose = verbose
        print(f'{name} Values Items Like So: ' + value_description)
        print('Total Inventory: ' + item_description  + '\n')

    def generate(self, message=''):
        if message is None and self.prompt == '' and self.description == '':
            pass
        else:
            message += self.prompt
            message += f" Recall that you are {self.description_dict[self.description]}."

        history = copy.deepcopy(self.history)
        history.append({"role":"user", "content":message})
        # pdb.set_trace()
        
        completion = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo",
            messages = history,
            temperature = 0.7,
            max_tokens = 256,
        )
        return(completion.choices[0].message.content)

    def add_message_to_history(self, message, sender='user'):
        self.history.append({"role":sender, "content":message})

