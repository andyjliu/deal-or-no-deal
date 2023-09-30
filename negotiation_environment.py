import os
import openai
import time
import random
import pdb
from negotiation_agent import NegotiationAgent

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

class NegotiationEnvironment():
    def __init__(self, logfile, eval_model='gpt-3.5-turbo', num_turns = 3):
        # TODO
        self.model = eval_model
        items = ['book', 'hat', 'ball']

        # [num_items, alice_val, bob_val]
        self.item_info = [random.choices(range(0,4), k=3) for i in range(3)]
        self.items = dict(zip(items, [i[0] for i in self.item_info]))
        self.alice_values = dict(zip(items, [i[1] for i in self.item_info]))
        self.bob_values = dict(zip(items, [i[2] for i in self.item_info]))

        self.agents = []
        self.agents.append(NegotiationAgent('Alice', 'Bob', num_turns, self.items, self.alice_values))
        self.agents.append(NegotiationAgent('Bob', 'Alice', num_turns, self.items, self.bob_values))

        self.total_turns = num_turns * len(self.agents)
        self.current_turn = 0
        self.max_attempts_per_round = 3
        self.message_history = [] # list of all messages
        self.proposal_history = [] # list of all proposals in standardized format
        self.reward_history = [] # list of rewards over time in form (A, B)
        self.logfile = logfile

    def standardize_proposal(self, proposal_msg):
        # TODO
        # convert a message into a standardized proposal format, e.g.
        # Alice: 1 book 2 hats 1 ball, Bob: 1 book 1 hat 3 balls.
        return('Alice: 1 book 2 hats 1 ball, Bob: 1 book 1 hat 3 balls.')

    def check_validity(self, proposal):
        # TODO
        # given a proposal, check to see if it has the correct number of total items. Return bool.
        return(True)

    def is_accepting(self, proposal):
        # TODO
        # given a proposal, check to see if it is accepting or proposal a counteroffer. Return bool.
        return(False)

    def compute_rewards(self, proposal):
        # return tuple (Alice's reward, Bob's reward) for a given proposal
        # TODO
        return((0,0))
        
    def step(self):
        next_agent = self.agents.pop(0) # get current agent

        num_attempts = 0
        next_message = next_agent.generate()
        standardized_proposal = self.standardize_proposal(next_message)
        while(not (self.check_validity(standardized_proposal) and num_attempts < self.max_attempts_per_round)):
            num_attempts += 1
            next_message = next_agent.generate()
            standardized_proposal = self.standardize_proposal(next_message)
        if num_attempts > self.max_attempts_per_round:
            raise AssertionError("Too Many Attempts to Generate Valid Proposal")

        self.message_history.append(next_message)
        self.proposal_history.append(standardized_proposal)
        self.reward_history.append(self.compute_rewards(standardized_proposal))

        self.agents.append(next_agent)
        self.current_turn += 1
        next_agent.add_message_to_history(next_message, sender='assistant')
        self.agents[0].add_message_to_history(f'{next_agent.name}\'s proposal: {standardized_proposal}')

        if self.current_turn >= self.total_turns or self.is_accepting(standardized_proposal):
            # game is over. log outputs and rewards
            to_log = ''
            assert(len(self.message_history) == len(self.proposal_history), "Mismatched lengths")
            for turn in range(len(self.message_history)):
                to_log += f"{self.message_history[turn]},"
                to_log += f"{self.proposal_history[turn]},"
                to_log += f"{self.reward_history[turn]},"
            # fill remaining columns (should be 18 total: 3 per turn for 6 turns)
            to_log += " ,"*(3*(self.total_turns - len(self.message_history)))
            # get rid of last comma and format
            to_log = to_log[:-1] 
            to_log += "\n"
            with open(self.logfile, 'w') as f:
                f.write(to_log)
            f.close()
            return None