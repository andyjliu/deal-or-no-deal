import pdb
import pandas as pd
import ast

def output_metrics(reward, items, alice_values, bob_values):
    # Fairness metric: exact difference between rewards of both agents
    numeric_rewards = [int(r) for r in reward if r.strip().isdigit()]
    difference = max(numeric_rewards) - min(numeric_rewards)
    # Fairness metric 2, avoiding unfairness in initial setup: (utility of agent 1)/(max agent 1 utility) - (utility of agent 2)/(max agent 2 utility) 
    # Calculate max utility of both agents
    alice_max_utility = 0
    bob_max_utility = 0

    for item in items.keys():
        alice_max_utility = alice_max_utility + (items[item] * alice_values[item])
    for item in items:
        bob_max_utility = bob_max_utility + (items[item] * bob_values[item])

    # print(f"alice max util: {alice_max_utility}")
    # print(f"bob max util: {bob_max_utility}")
    # print(reward[4])
    total_utility_percentage_fairness = int(reward[1])/alice_max_utility - int(reward[4])/bob_max_utility
    welfare = (int(reward[1]) + int(reward[4]))/(alice_max_utility + bob_max_utility)
    return({'Absolute Fairness':difference, 'Normalized Fairness':total_utility_percentage_fairness, 'Welfare':welfare,
            'A Utility':int(reward[1])/alice_max_utility, 'B Utility':int(reward[4])/bob_max_utility})

def summary_from_logfile(logfile):
    df = pd.read_csv(logfile)
    to_eval = []

    accept = []
    abs_fairness = []
    normal_fairness = []
    welfare = []
    a_util = []
    b_util = []

    for index, row in df.iterrows(): 
        if len(row['Item Quantities']) > 1:
            items = ast.literal_eval(row['Item Quantities'])
            alice_values = ast.literal_eval(row['A Values'])
            bob_values = ast.literal_eval(row['B Values'])
            rewards = []

            #pdb.set_trace()
            accepted = False
            turns = ['B3', 'A3', 'B2', 'A2', 'B1', 'A1']
            for turn in turns:
                next_offer = row[turn + '  Offer']
                if type(next_offer) is str and 'accept' in next_offer.lower():
                    accepted = True
            accept.append(accepted)

            for turn in turns:
                next_reward = row[turn + '  Rewards']
                #pdb.set_trace()
                if type(next_reward) is str:
                    reward = next_reward
                    to_eval.append((reward, items, alice_values, bob_values))
                    break

    for eval in to_eval:
        try:
            metrics = output_metrics(*eval)
            abs_fairness.append(float(metrics['Absolute Fairness']))
            normal_fairness.append(float(metrics['Normalized Fairness']))
            welfare.append(float(metrics['Welfare']))
            a_util.append(float(metrics['A Utility']))
            b_util.append(float(metrics['B Utility']))

        except ZeroDivisionError:
            pass

    print(f'Acceptance Rate: {sum(accept)/len(accept)}\n \
          Average Fairness: {sum(abs_fairness)/len(abs_fairness)}\n \
          Average Normalized Fairness: {sum(normal_fairness)/len(normal_fairness)}\n \
          Average Normalized Welfare: {sum(welfare)/len(welfare)}\n \
          Average Alice Utility: {sum(a_util)/len(a_util)}\n \
          Average Bob Utility: {sum(b_util)/len(b_util)}\n'
          )