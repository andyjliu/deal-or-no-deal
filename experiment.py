import argparse
import time
import os
import openai
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

    parser.add_argument('--seed', type=int, default=0, help='random seed for reproducibility')
    parser.add_argument('--output', type=str, default=f'test_{time.strftime("%Y%m%d-%H%M%S")}.csv',
        help = 'Name of output csv')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    env = NegotiationEnvironment(logfile=args.output, a_desc=args.a_desc, b_desc=args.b_desc,
                                 a_prompt=args.a_prompt, b_prompt=args.b_prompt, 
                                 eval_model='gpt-3.5-turbo', num_turns=3, seed=args.seed)
    for iter in range(args.num_iters):
        is_complete = False
        try:
            is_complete = False
            while not is_complete:
                is_complete = env.step()
        except AssertionError or openai.error.OpenAIError as e:
            print(f'Error {e} on Iteration {iter}')