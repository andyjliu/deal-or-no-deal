import os
import re
import openai
import random
import time
import pdb
from negotiation_agent import NegotiationAgent

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

class NegotiationEnvironment():
    def __init__(self, logfile, 
                 a_desc='default', b_desc='default',
                 a_prompt='CoT', b_prompt='CoT',
                 eval_model='gpt-3.5-turbo', num_turns = 3, seed = 0):
        self.model = eval_model
        items = ['book', 'hat', 'ball']

        # [num_items, alice_val, bob_val]
        random.seed(seed)
        self.item_info = [random.choices(range(0,4), k=3) for i in range(3)]
        self.items = dict(zip(items, [i[0] for i in self.item_info]))
        self.alice_values = dict(zip(items, [i[1] for i in self.item_info]))
        self.bob_values = dict(zip(items, [i[2] for i in self.item_info]))

        self.agents = []
        self.agents.append(NegotiationAgent('Alice', 'Bob', num_turns, self.items, 
                                            self.alice_values, a_desc, a_prompt))
        self.agents.append(NegotiationAgent('Bob', 'Alice', num_turns, self.items, 
                                            self.bob_values, b_desc, b_prompt))

        self.total_turns = num_turns * len(self.agents)
        self.current_turn = 0
        self.max_attempts_per_round = 3
        self.message_history = [] # list of all messages
        self.proposal_history = [] # list of all proposals in standardized format
        self.reward_history = [] # list of rewards over time in form (A, B)
        self.logfile = logfile

    def word_to_number(self, word):
        word_to_num = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'zero': 0,
        }
        return word_to_num.get(word, word)

    def standardize_proposal(self, proposal_msg, next_agent):
        current_agent_name = next_agent.name
        print(f'______________________{current_agent_name}______________________')
        print(f"Original generated proposal: {proposal_msg}")
        opp_agent_name = 'Bob' if current_agent_name.lower() == 'alice' else 'Alice' 

        prompt = f"Message: This proposed deal gives me a total value of (2x4 + 4x1) 12. Even though the two books are of lesser value to me, I still think I can carry out a negotiation where we both can get more value. I would like to propose a new deal: 'I get 1 book, 3 hats, and 2 balls.'\nOffer: '1 book 3 hats 2 ball'\nMessage: {proposal_msg}\nOffer:"
        response = openai.ChatCompletion.create(
            model = self.model,
            messages = [{"role": "user", "content": prompt}],
            temperature = 0.7,
            max_tokens = 15,
        )
        generated_offer = response.choices[0].message.content.strip()
        cleaned_generated_offer = generated_offer.strip().replace("'", "").replace("\"", "")
        
        # Split out the portion of the cleaned_generated_offer, if both names exist
        if (opp_agent_name in cleaned_generated_offer) and (current_agent_name in cleaned_generated_offer):
            cleaned_generated_offer = cleaned_generated_offer.split(opp_agent_name)[0].strip()

        # Extract the counts from current agent's offer
        items_counts = {
            'book': 0,
            'hat': 0,
            'ball': 0
        }
        patterns = [
            (r"(\d+) books?", 'book'),
            (r"(\d+) hats?", 'hat'),
            (r"(\d+) balls?", 'ball')
        ]
        for pattern, item in patterns:
            match = re.search(pattern, cleaned_generated_offer)
            if match:
                items_counts[item] = int(match.group(1))

        # Format current agent's offer in a standardized way
        cleaned_generated_offer_standardized = f"{items_counts.get('book', 0)} book {items_counts.get('hat', 0)} hat {items_counts.get('ball', 0)} ball"
        print(f"Offer from {current_agent_name}: {cleaned_generated_offer_standardized}")

        # Get count for opponent agent
        opp_items_counts = {}
        for item, count in items_counts.items():
            opp_items_counts[item] = self.items[item] - count

        remaining_offer = f"{opp_agent_name}: {opp_items_counts.get('book', 0)} book {opp_items_counts.get('hat', 0)} hat {opp_items_counts.get('ball', 0)} ball"
        standardized_proposal = f"'{current_agent_name}: {cleaned_generated_offer_standardized} {remaining_offer}'"
        print(f"Standardized Proposal: {standardized_proposal}")

        return standardized_proposal


    def check_validity(self, proposal):
        items_pattern = r"(\w+): (\w+|\d+) (book|ball|hat)s? (\w+|\d+) (book|ball|hat)s? (\w+|\d+) (book|ball|hat)s?"
        matches = re.findall(items_pattern, proposal)

        if len(matches) != 2:  # change this if we want to allow more than 2 agents
            return False

        total_counts = {}
        for match in matches:
            for i in range(1, 7, 2):
                item_count = self.word_to_number(match[i])
                item_name = match[i+1]
                total_counts[item_name] = total_counts.get(item_name, 0) + int(item_count)

        for item, count in self.items.items():
            if total_counts.get(item, 0) != count:
                return False

        return True

    def is_accepting(self, proposal):
        proposal = proposal.lower()
        acceptance_terms = ['accepted', 'agree', 'okay', 'accept']

        # Check if any of acceptance terms above is in the proposal
        if any(term in proposal for term in acceptance_terms):
            return True
        return False

    def compute_rewards(self, proposal):
        # Extracting counts of each item for Alice and Bob from the proposal
        items_pattern = r"(\w+): (\w+|\d+) (book|ball|hat)s? (\w+|\d+) (book|ball|hat)s? (\w+|\d+) (book|ball|hat)s?"
        matches = re.findall(items_pattern, proposal)

        if len(matches) != 2:  
            return (0, 0)

        alice_items, bob_items = matches

        # Utility function to compute reward for an agent based on item counts and their values
        def compute_individual_reward(items, values):
            reward = 0
            for i in range(1, 7, 2):
                count = int(self.word_to_number(items[i]))
                item = items[i+1]
                reward += count * values[item]
            return reward

        alice_reward = compute_individual_reward(alice_items, self.alice_values)
        bob_reward = compute_individual_reward(bob_items, self.bob_values)

        return (alice_reward, bob_reward)
        
    def step(self):
        # plays one round in negotiation game.
        # returns True if more moves can be made.

        next_agent = self.agents.pop(0) # get current agent

        num_attempts = 0
        turn_dict = {1:'first', 2:'second', 3:'third'}
        next_message = next_agent.generate(message=f'It is your turn to make the {turn_dict[self.current_turn//2 + 1]} offer, {next_agent.name}.')
        print(f"NEXT MESSAGE: {next_message}")
        standardized_proposal = self.standardize_proposal(next_message, next_agent)
        # pdb.set_trace()
        while(not (self.check_validity(standardized_proposal) and num_attempts < self.max_attempts_per_round)):
            num_attempts += 1
            next_message = next_agent.generate()
            standardized_proposal = self.standardize_proposal(next_message, next_agent)
            time.sleep(5)

        if num_attempts > self.max_attempts_per_round:
            raise AssertionError("Too Many Attempts to Generate Valid Proposal")

        self.message_history.append(next_message)
        self.proposal_history.append(standardized_proposal)
        self.reward_history.append(str(self.compute_rewards(standardized_proposal)))

        self.agents.append(next_agent)
        self.current_turn += 1
        next_agent.add_message_to_history(next_message, sender='assistant')
        self.agents[0].add_message_to_history(f'{next_agent.name}\'s proposal: {standardized_proposal}')

        if self.current_turn >= self.total_turns or self.is_accepting(next_message):
            # game is over. log outputs and rewards
            self.proposal_history[-1] = "Accept"
            assert len(self.message_history) == len(self.proposal_history), "Mismatched lengths"
            to_log = []
            for item1, item2, item3 in zip(self.message_history, self.proposal_history, self.reward_history):
                to_log.extend([item1, item2, item3])
            to_log = "|".join(to_log)
            with open(self.logfile, 'w', newline='') as f:
                f.write(to_log)
                f.write('\n')
            f.close()
            return(True)
        
        else:
            return(False)
        
    def reset(self):
        # resets environment while maintaining values and item counts
        num_turns = self.total_turns/len(self.agents)
        self.agents = []
        self.agents.append(NegotiationAgent('Alice', 'Bob', num_turns, self.items, self.alice_values))
        self.agents.append(NegotiationAgent('Bob', 'Alice', num_turns, self.items, self.bob_values))
        
        self.current_turn = 0
        self.message_history = [] 
        self.proposal_history = [] 
        self.reward_history = [] 

        return(None)