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
MEMORY_CAPACITY = 2000                          # 记忆库容量
device='cuda' if torch.cuda.is_available() else 'cpu'
class ReplayBuffer():
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        idx = self.mem_cntr % self.mem_size

        self.state_memory[idx] = state
        self.new_state_memory[idx] = state_
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.terminal_memory[idx] = done

        self.mem_cntr += 1

    def sample(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones


class DuelingDeepQNet(nn.Module):
    def __init__(self, n_actions, input_dim, fc1_dims, fc2_dims, lr=0.0003):
        super(DuelingDeepQNet, self).__init__()

        self.fc1 = nn.Linear(*input_dim, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.V = nn.Linear(fc2_dims, 1)
        self.A = nn.Linear(fc2_dims, n_actions)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        self.optim = optim.Adam(self.parameters(), lr=lr)
        self.crit = nn.MSELoss()

    def forward(self, state):
        x = self.relu1(self.fc1(state))
        x = self.relu2(self.fc2(x))

        V = self.V(x)
        A = self.A(x)

        Q = V + (A - torch.mean(A, dim=1, keepdim=True))

        return Q

    def advantage(self, state):
        x = self.relu1(self.fc1(state))
        x = self.relu2(self.fc2(x))

        return self.A(x)


class Agent:
    def __init__(self, gamma, n_actions, epsilon, batch_size,
                 input_dims, epsilon_decay=1e-8, eps_min=0.01,
                 mem_size=1000000, fc1_dims=128, fc2_dims=128, replace=100):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.eps_min = eps_min
        self.replace = replace
        self.batch_size = batch_size
        '''self.q_eval = torch.load('../param/net_param/q_eval_D3QN.pkl')
        self.q_eval.eval()
        self.q_next = torch.load('../param/net_param/q_next_D3QN.pkl')
        self.q_next.eval()'''
        self.learn_step_counter = 0
        self.memory = ReplayBuffer(max_size=mem_size, input_shape=input_dims)
        self.q_eval = DuelingDeepQNet(n_actions=n_actions, input_dim=input_dims, fc1_dims=fc1_dims, fc2_dims=fc2_dims)
        self.q_next = DuelingDeepQNet(n_actions=n_actions, input_dim=input_dims, fc1_dims=fc1_dims, fc2_dims=fc2_dims)
        if not os.path.exists('../param'):
            os.makedirs('../param/net_param')
        self.q_eval.to(device)
        self.q_next.to(device)

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            state = torch.Tensor([observation]).to(device)
            advantage = self.q_eval.advantage(state)
            action = torch.argmax(advantage).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        if self.learn_step_counter % self.replace == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

        states, actions, rewards, states_, dones = self.memory.sample(self.batch_size)

        states = torch.tensor(states).to(device)
        rewards = torch.tensor(rewards).to(device)
        dones = torch.tensor(dones).to(device)
        actions = torch.tensor(actions).to(device)
        states_ = torch.tensor(states_).to(device)

        indices = np.arange(self.batch_size)

        q_pred = self.q_eval(states)[indices, actions]
        q_next = self.q_next(states_)

        max_actions = torch.argmax(self.q_eval(states_), dim=1)
        q_target = rewards + self.gamma * q_next[indices, max_actions]

        q_next[dones] = 0.0
        self.q_eval.optim.zero_grad()

        loss = self.q_eval.crit(q_target, q_pred)
        loss.backward()

        self.q_eval.optim.step()

        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.eps_min else self.eps_min
        self.learn_step_counter += 1

def main():
    env = football_env.create_environment(
      env_name='academy_empty_goal_close',
      stacked=False,
      representation='simple115v2',
      rewards = 'scoring',
      write_goal_dumps=False,
      write_full_episode_dumps=False,
      dump_frequency = 0)
    torch.manual_seed(seed)
    env.seed(seed)
    obs_shape = list(env.observation_space.shape)
    agent = Agent(gamma=0.99, n_actions=num_action, epsilon=0.99, batch_size=64, input_dims=obs_shape)
    win=0
    lost=0
    for i_epoch in range(1000000):
        if i_epoch%10==0:
            torch.save(agent.q_eval, '../param/net_param/q_eval_D3QN.pkl')
            torch.save(agent.q_next, '../param/net_param/q_next_D3QN.pkl')
        state = env.reset()
        step = 0
        sum = 0.0
        flag = 0
        numof4=0
        while True:
            step+=1
            if render: env.render()
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            sum+=reward
            agent.store_transition(state, action, reward, next_state,int(done))
            state = next_state
            agent.learn()
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



