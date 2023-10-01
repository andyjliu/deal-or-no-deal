import argparse
import time
import os
import openai
import pdb
import csv
from negotiation_environment import NegotiationEnvironment
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num-iters', type=int, default=20,
        help = 'number of times to play Deal or No Deal')
    parser.add_argument('--num-rounds', type=int, default=3,
        help = 'Number of negotiation rounds per game of Deal or No Deal')
    
    parser.add_argument('--a-desc', type=str, default='default',
        help = 'personality type of agent A')
    parser.add_argument('--b-desc', type=str, default='default',
        help = 'personality type of agent B')
    parser.add_argument('--a-prompt', type=str, default='CoT',
        help = 'type of prompt to use for agent A')
    parser.add_argument('--b-prompt', type=str, default='CoT',
        help = 'type of prompt to use for agent B')
    parser.add_argument('--agent-model', type=str, default='gpt-4',
        help = 'Base model to use for the agents')
    parser.add_argument('--eval-model', type=str, default='gpt-3.5-turbo',
        help = 'Base model to use for eval')
    parser.add_argument('--conversational', type=bool, default=False,
        help = 'Set to True if the agents should communicate in a conversational way,'
               'by sharing their reasoning. Set to False if they can only communicate standardized numeric proposals')

    # parser.add_argument('--seed', type=int, default=0, help='random seed for reproducibility')
    parser.add_argument('--output', type=str, default=f'test_{time.strftime("%Y%m%d-%H%M%S")}.csv',
        help = 'Name of output csv')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    env = NegotiationEnvironment(logfile=args.output, a_desc=args.a_desc, b_desc=args.b_desc,
                                 a_prompt=args.a_prompt, b_prompt=args.b_prompt, 
                                 eval_model=args.eval_model,
                                 agent_model=args.agent_model,
                                 num_turns=args.num_rounds, verbose=args.verbose,
                                 conversational=args.conversational
                                )
    # init log file
    with open(args.output, 'w') as f:
        to_log = ['Item Quantities', 'A Values', 'B Values']
        cols = ['Message', 'Offer', 'Rewards']
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        for n in range(2*args.num_rounds):
            turn_str = f"{['A', 'B'][n % 2]}{n//2 + 1} "
            to_log.extend([f"{turn_str} {col}" for col in cols])
        wr.writerow(to_log)

    for iter in range(args.num_iters):
        is_complete = False
        try:
            is_complete = False
            while not is_complete:
                is_complete = env.step()
            print(f'Completed Iteration {iter}')
        except AssertionError or openai.error.OpenAIError as e:
            print(f'Error {e} on Iteration {iter}')