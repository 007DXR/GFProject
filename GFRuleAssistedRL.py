import argparse
import pickle
from collections import namedtuple
from itertools import count
import gfootball.env as football_env
import os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

# Parameters
gamma = 0.99
render =False
seed = 1
log_interval = 10
INTVAL=10
num_state = 115
num_action = 19

Transition = namedtuple('Transition', ['state', 'action',  'a_log_prob', 'reward', 'next_state'])
#print(num_action)
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_state, 128)
        self.action_head = nn.Linear(128, num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_state, 128)
        self.state_value = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = self.state_value(x)
        return value


class PPO():
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_update_time = 10
    buffer_capacity = 1000
    batch_size = 32

    def __init__(self):
        super(PPO, self).__init__()
        self.actor_net = Actor()
        self.critic_net = Critic()
        #self.actor_net = torch.load('../param/net_param/actor_netRuleAssistedRL.pkl')
        #self.actor_net.eval()
        #self.critic_net = torch.load('../param/net_param/critic_netRuleAssistedRL.pkl')
        #self.critic_net.eval()
        
        self.buffer = []
        self.counter = 0
        self.training_step = 0

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), 1e-3)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), 3e-3)
        if not os.path.exists('../param'):
            os.makedirs('../param/net_param')
    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_prob = self.actor_net(state)
        c = Categorical(action_prob)
        action = c.sample()
        return action.item(), action_prob[:,action.item()].item()

    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()
    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1


    def update(self, i_ep):
        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1)
        reward = [t.reward for t in self.buffer]
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1)

        R = 0
        Gt = []
        for r in reward[::-1]:
            R = r + gamma * R
            Gt.insert(0, R)
        Gt = torch.tensor(Gt, dtype=torch.float)
        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):
                Gt_index = Gt[index].view(-1, 1)
                V = self.critic_net(state[index])
                delta = Gt_index - V
                advantage = delta.detach()
                # epoch iteration, PPO core!!!
                action_prob = self.actor_net(state[index]).gather(1, action[index]) # new policy

                ratio = (action_prob/old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                # update actor network
                action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                #update critic network
                value_loss = F.mse_loss(Gt_index, V)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                self.training_step += 1

        del self.buffer[:] # clear experience
def flt(step,action):
    if 0<step<25: return 3
    if step==25: return 7
    if step==26: return 11
    if 26<step<45 and (action<3 or action >7): return 5
    return action
    
def main():
    env = football_env.create_environment(
      env_name='academy_3_vs_1_with_keeper',
      stacked=False,
      representation='simple115v2',
      rewards = 'scoring',
      write_goal_dumps=False,
      write_full_episode_dumps=False,
      dump_frequency = 0)
    torch.manual_seed(seed)
    env.seed(seed)
    agent = PPO()
    n_actions = env.action_space.n
    obs_shape = list(env.observation_space.shape)
    win=0
    lost=0
    for i_epoch in range(1000000):
        if i_epoch%10==0:
            torch.save(agent.actor_net, '../param/net_param/actor_netRuleAssistedRL.pkl')
            torch.save(agent.critic_net, '../param/net_param/critic_netRuleAssistedRL.pkl')
        state = env.reset()
        step = 0
        sum = 0.0
        flag = 0
        numof4=0
        while True:
            step+=1
            if render: env.render()
            action, action_prob = agent.select_action(state)
            action = flt(step,action)
            next_state, reward, done, _ = env.step(action)
            sum+=reward
            agent.store_transition(Transition(state, action, action_prob, reward, next_state))
            state = next_state
            if len(agent.buffer) >= agent.batch_size:
                 agent.update(i_epoch)
            if done:
                break

        winbef=win
        lostbef=lost
        if sum==1:win+=1
        if sum==-1:lost+=1
        print("{}\t{}\t{}\t{}".format(i_epoch,win,step,lost))

if __name__ == '__main__':
    main()
    
    print("end")



