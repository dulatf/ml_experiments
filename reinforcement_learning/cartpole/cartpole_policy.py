import argparse
import gym
from gym.wrappers import Monitor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from itertools import count

eps = np.finfo(np.float32).eps.item()


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.linear1 = nn.Linear(4,32) 
        self.linear2 = nn.Linear(32, 32)
        self.linear3 = nn.Linear(32, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        action_scores = self.linear3(x)
        return F.softmax(action_scores, dim=1)


def select_action(policy, state,eval_mode=False):
    """Evaluate policy network and stochastically select action based on the result"""
    # turn input n-vector into 1xn pytorch tensor
    state = torch.from_numpy(state).float().unsqueeze(0)
    # run our policy network for the current state
    probs = policy(state)
    # creates a categorical distribution
    # (binomial here, since there are only two possible actions
    dist = Categorical(probs)
    # sample from the distribution to stochastically pick the action
    action = dist.sample()
    if not eval_mode:
        policy.saved_log_probs.append(dist.log_prob(action))
    return action.item()


def update_policy(optimizer, policy, args):
    R = 0
    policy_loss = []
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]

def evaluate_statistics(env, policy):
    print("Gathering statistics...")
    scores = []
    for i in range(100):
        state = env.reset()
        for t in count(1):
            action = select_action(policy,state,eval_mode=True)
            state, reward, done, _ = env.step(action)
            if done:
                break
        scores.append(t)
    scores = torch.tensor(scores).float()
    print("Mean: {}, StDev: {}".format(scores.mean(),scores.std()))


def run(optimizer, policy, env, args):
    running_reward = 10
    for i_episode in count(1):
        state = env.reset()
        for t in range(10000):
            action = select_action(policy, state)
            state, reward, done, _ = env.step(action)
            if args.render:
                env.render()
            policy.rewards.append(reward)
            if done:
                break

        running_reward = running_reward * 0.99 + t * 0.01
        update_policy(optimizer, policy, args)
        if i_episode % args.log_interval == 0:
            print("Episode {}\t Last length: {:5d}\tAverage length: {:.2f}".format(i_episode, t, running_reward))
            with torch.no_grad():
                evaluate_statistics(env,policy)
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode ran o {} time steps!".format(running_reward, t))
            print("Saving model...")
            torch.save(policy.state_dict(),"./model_weights")
            break


def main(env, args):
    policy = Policy()
    torch.save(policy.state_dict(),"./model_weights")
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    if args.render:
        env.render()
    run(optimizer, policy, env, args)


def playback(env, args):
    policy = Policy()
    policy.load_state_dict(torch.load('./model_weights'))
#    env = Monitor(env, './video',force=True)
    evaluate_statistics(env, policy)
    env.render()
    state = env.reset()
    for t in count(1):
        action = select_action(policy, state)
        state, reward, done, _ = env.step(action)
        env.render()
        if done:
            break
    print("Finished after {} episodes.\nBye!".format(t))
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cartpole with PyTorch')
    parser.add_argument('--gamma', type=float, default=0.99,
                        metavar='G', help='discount factor [0.99]')
    parser.add_argument('--render', action='store_true',
                        help='render the simulation')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='interval between training status logs [10]')
    parser.add_argument('--playback', action='store_true',help='load the model and render one episode')
    args = parser.parse_args()
    env = gym.make('CartPole-v0')
    if args.playback:
        playback(env,args)
    else:
        main(env, args)
