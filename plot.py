import matplotlib.pyplot as plt
import json
import ast
from pathlib import Path

def plot_welfare(env, results_dir):
  figfile = Path(results_dir, 'welfare.png')
  rewards = ([ast.literal_eval(reward_step) for reward_step in env.reward_history])
  welfare = [sum(reward) for reward in rewards]
  fig, ax = plt.subplots()
  ax.plot(list(range(len(welfare))), welfare)
  ax.set_xlabel('Turn Number')
  ax.set_ylabel('Welfare')
  ax.set_title('Welfare Over Time')
  fig.savefig(figfile)

def plot_fairness(env, results_dir):
  figfile = Path(results_dir, 'fairness.png')
  rewards = ([ast.literal_eval(reward_step) for reward_step in env.reward_history])
  fairness = [reward[0] - reward[1] for reward in rewards]
  fig, ax = plt.subplots()
  ax.plot(list(range(len(fairness))), fairness)
  ax.set_xlabel('Turn Number')
  ax.set_ylabel('Fairness')
  ax.set_title('Fairness Over Time')
  fig.savefig(figfile)
