import gfootball.env as football_env
import os
import argparse
import numpy as np
import torch
from torch import optim
from datetime import datetime
import copy
from torch import nn
from torch.nn import functional as F
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from torch.distributions.categorical import Categorical
import random
import logging

# select actions
def select_actions(pi):
    actions = Categorical(pi).sample()
    # return actions
    return actions.detach().cpu().numpy().squeeze()

# evaluate actions
def evaluate_actions(pi, actions):
    cate_dist = Categorical(pi)
    log_prob = cate_dist.log_prob(actions).unsqueeze(-1)
    entropy = cate_dist.entropy().mean()
    return log_prob, entropy

# configure the logger
def config_logger(log_dir):
    logger = logging.getLogger()
    # we don't do the debug...
    logger.setLevel('INFO')
    basic_format = '%(message)s'
    formatter = logging.Formatter(basic_format)
    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)
    # set the log file handler
    fhlr = logging.FileHandler(log_dir)
    logger.addHandler(chlr)
    logger.addHandler(fhlr)
    return logger
def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--gamma', type=float, default=0.993, help='the discount factor of RL')
    parse.add_argument('--seed', type=int, default=123, help='the random seeds')
    parse.add_argument('--num-workers', type=int, default=8, help='the number of workers to collect samples')
    parse.add_argument('--env-name', type=str, default='academy_3_vs_1_with_keeper', help='the environment name')
    parse.add_argument('--batch-size', type=int, default=8, help='the batch size of updating')
    parse.add_argument('--lr', type=float, default=0.00008, help='learning rate of the algorithm')
    parse.add_argument('--epoch', type=int, default=4, help='the epoch during training')
    parse.add_argument('--nsteps', type=int, default=128, help='the steps to collect samples')
    parse.add_argument('--vloss-coef', type=float, default=0.5, help='the coefficient of value loss')
    parse.add_argument('--ent-coef', type=float, default=0.01, help='the entropy loss coefficient')
    parse.add_argument('--tau', type=float, default=0.95, help='gae coefficient')
    parse.add_argument('--cuda', action='store_true', help='use cuda do the training')
    parse.add_argument('--total-frames', type=int, default=int(2e8), help='the total frames for training')
    parse.add_argument('--eps', type=float, default=1e-5, help='param for adam optimizer')
    parse.add_argument('--clip', type=float, default=0.27, help='the ratio clip param')
    parse.add_argument('--save-dir', type=str, default='./', help='the folder to save models')
    parse.add_argument('--lr-decay', action='store_true', help='if using the learning rate decay during decay')
    parse.add_argument('--max-grad-norm', type=float, default=0.5, help='grad norm')
    parse.add_argument('--display-interval', type=int, default=10, help='the interval that display log information')
    parse.add_argument('--log-dir', type=str, default='logs/')

    args = parse.parse_args()

    return args



class ppo_agent:
    def __init__(self, envs, args, net):
        self.envs = envs 
        self.args = args
        # define the newtork...
        self.net = net
        self.old_net = copy.deepcopy(self.net)
        # if use the cuda...
        if self.args.cuda:
            self.net.cuda()
            self.old_net.cuda()
        # define the optimizer...
        self.optimizer = optim.Adam(self.net.parameters(), self.args.lr, eps=self.args.eps)
        # check saving folder..
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # env folder..
        self.model_path = os.path.join(self.args.save_dir, self.args.env_name)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        # logger folder
        if not os.path.exists(self.args.log_dir):
            os.mkdir(self.args.log_dir)
        self.log_path = self.args.log_dir + self.args.env_name + '.log'
        # get the observation
        self.batch_ob_shape = (self.args.num_workers * self.args.nsteps, ) + self.envs.observation_space.shape
        self.obs = np.zeros((self.args.num_workers, ) + self.envs.observation_space.shape, dtype=self.envs.observation_space.dtype.name)
        self.obs[:] = self.envs.reset()
        self.dones = [False for _ in range(self.args.num_workers)]
        self.logger = config_logger(self.log_path)

    # start to train the network...
    def learn(self):
        num_updates = self.args.total_frames // (self.args.nsteps * self.args.num_workers)
        # get the reward to calculate other informations
        episode_rewards = torch.zeros([self.args.num_workers, 1])
        final_rewards = torch.zeros([self.args.num_workers, 1])
        for update in range(num_updates):
            mb_obs, mb_rewards, mb_actions, mb_dones, mb_values = [], [], [], [], []
            if self.args.lr_decay:
                self._adjust_learning_rate(update, num_updates)
            for step in range(self.args.nsteps):
                with torch.no_grad():
                    # get tensors
                    obs_tensor = self._get_tensors(self.obs)
                    values, pis = self.net(obs_tensor)
                # select actions
                actions = select_actions(pis)
                # get the input actions
                input_actions = actions 
                # start to store information
                mb_obs.append(np.copy(self.obs))
                mb_actions.append(actions)
                mb_dones.append(self.dones)
                mb_values.append(values.detach().cpu().numpy().squeeze())
                # start to excute the actions in the environment
                obs, rewards, dones, _ = self.envs.step(input_actions)
                # update dones
                self.dones = dones
                mb_rewards.append(rewards)
                # clear the observation
                for n, done in enumerate(dones):
                    if done:
                        self.obs[n] = self.obs[n] * 0
                self.obs = obs
                # process the rewards part -- display the rewards on the screen
                rewards = torch.tensor(np.expand_dims(np.stack(rewards), 1), dtype=torch.float32)
                episode_rewards += rewards
                masks = torch.tensor([[0.0] if done_ else [1.0] for done_ in dones], dtype=torch.float32)
                final_rewards *= masks
                final_rewards += (1 - masks) * episode_rewards
                episode_rewards *= masks
            # process the rollouts
            mb_obs = np.asarray(mb_obs, dtype=np.float32)
            mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
            mb_actions = np.asarray(mb_actions, dtype=np.float32)
            mb_dones = np.asarray(mb_dones, dtype=np.bool)
            mb_values = np.asarray(mb_values, dtype=np.float32)
            # compute the last state value
            with torch.no_grad():
                obs_tensor = self._get_tensors(self.obs)
                last_values, _ = self.net(obs_tensor)
                last_values = last_values.detach().cpu().numpy().squeeze()
            # start to compute advantages...
            mb_returns = np.zeros_like(mb_rewards)
            mb_advs = np.zeros_like(mb_rewards)
            lastgaelam = 0
            for t in reversed(range(self.args.nsteps)):
                if t == self.args.nsteps - 1:
                    nextnonterminal = 1.0 - self.dones
                    nextvalues = last_values
                else:
                    nextnonterminal = 1.0 - mb_dones[t + 1]
                    nextvalues = mb_values[t + 1]
                delta = mb_rewards[t] + self.args.gamma * nextvalues * nextnonterminal - mb_values[t]
                mb_advs[t] = lastgaelam = delta + self.args.gamma * self.args.tau * nextnonterminal * lastgaelam
            mb_returns = mb_advs + mb_values
            # after compute the returns, let's process the rollouts
            mb_obs = mb_obs.swapaxes(0, 1).reshape(self.batch_ob_shape)
            mb_actions = mb_actions.swapaxes(0, 1).flatten()
            mb_returns = mb_returns.swapaxes(0, 1).flatten()
            mb_advs = mb_advs.swapaxes(0, 1).flatten()
            # before update the network, the old network will try to load the weights
            self.old_net.load_state_dict(self.net.state_dict())
            # start to update the network
            pl, vl, ent = self._update_network(mb_obs, mb_actions, mb_returns, mb_advs)
            # display the training information
            if update % self.args.display_interval == 0:
                self.logger.info('[{}] Update: {} / {}, Frames: {}, Rewards: {:.3f}, Min: {:.3f}, Max: {:.3f}, PL: {:.3f},'\
                    'VL: {:.3f}, Ent: {:.3f}'.format(datetime.now(), update, num_updates, (update + 1)*self.args.nsteps*self.args.num_workers, \
                    final_rewards.mean().item(), final_rewards.min().item(), final_rewards.max().item(), pl, vl, ent))
                # save the model
                torch.save(self.net.state_dict(), self.model_path + '/model.pt')

    # update the network
    def _update_network(self, obs, actions, returns, advantages):
        inds = np.arange(obs.shape[0])
        nbatch_train = obs.shape[0] // self.args.batch_size
        for _ in range(self.args.epoch):
            np.random.shuffle(inds)
            for start in range(0, obs.shape[0], nbatch_train):
                # get the mini-batchs
                end = start + nbatch_train
                mbinds = inds[start:end]
                mb_obs = obs[mbinds]
                mb_actions = actions[mbinds]
                mb_returns = returns[mbinds]
                mb_advs = advantages[mbinds]
                # convert minibatches to tensor
                mb_obs = self._get_tensors(mb_obs)
                mb_actions = torch.tensor(mb_actions, dtype=torch.float32)
                mb_returns = torch.tensor(mb_returns, dtype=torch.float32).unsqueeze(1)
                mb_advs = torch.tensor(mb_advs, dtype=torch.float32).unsqueeze(1)
                # normalize adv
                mb_advs = (mb_advs - mb_advs.mean()) / (mb_advs.std() + 1e-8)
                if self.args.cuda:
                    mb_actions = mb_actions.cuda()
                    mb_returns = mb_returns.cuda()
                    mb_advs = mb_advs.cuda()
                # start to get values
                mb_values, pis = self.net(mb_obs)
                # start to calculate the value loss...
                value_loss = (mb_returns - mb_values).pow(2).mean()
                # start to calculate the policy loss
                with torch.no_grad():
                    _, old_pis = self.old_net(mb_obs)
                    # get the old log probs
                    old_log_prob, _ = evaluate_actions(old_pis, mb_actions)
                    old_log_prob = old_log_prob.detach()
                # evaluate the current policy
                log_prob, ent_loss = evaluate_actions(pis, mb_actions)
                prob_ratio = torch.exp(log_prob - old_log_prob)
                # surr1
                surr1 = prob_ratio * mb_advs
                surr2 = torch.clamp(prob_ratio, 1 - self.args.clip, 1 + self.args.clip) * mb_advs
                policy_loss = -torch.min(surr1, surr2).mean()
                # final total loss
                total_loss = policy_loss + self.args.vloss_coef * value_loss - ent_loss * self.args.ent_coef
                # clear the grad buffer
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.max_grad_norm)
                # update
                self.optimizer.step()
        return policy_loss.item(), value_loss.item(), ent_loss.item()

    # convert the numpy array to tensors
    def _get_tensors(self, obs):
        obs_tensor = torch.tensor(np.transpose(obs, (0, 3, 1, 2)), dtype=torch.float32)
        # decide if put the tensor on the GPU
        if self.args.cuda:
            obs_tensor = obs_tensor.cuda()
        return obs_tensor

    # adjust the learning rate
    def _adjust_learning_rate(self, update, num_updates):
        lr_frac = 1 - (update / num_updates)
        adjust_lr = self.args.lr * lr_frac
        for param_group in self.optimizer.param_groups:
             param_group['lr'] = adjust_lr

# create the environment
def create_single_football_env(args):
    """Creates gfootball environment."""
    env = football_env.create_environment(\
            env_name=args.env_name, stacked=True,render=False
            )
    return env


"""
this network is modified for the google football

"""
# the convolution layer of deepmind
class deepmind(nn.Module):
    def __init__(self):
        super(deepmind, self).__init__()
        self.conv1 = nn.Conv2d(16, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1)
        self.fc1 = nn.Linear(32 * 5 * 8, 512)        
        # start to do the init...
        nn.init.orthogonal_(self.conv1.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.conv2.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.conv3.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.fc1.weight.data, gain=nn.init.calculate_gain('relu'))
        # init the bias...
        nn.init.constant_(self.conv1.bias.data, 0)
        nn.init.constant_(self.conv2.bias.data, 0)
        nn.init.constant_(self.conv3.bias.data, 0)
        nn.init.constant_(self.fc1.bias.data, 0)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 32 * 5 * 8)
        x = F.relu(self.fc1(x))

        return x

# in the initial, just the nature CNN
class cnn_net(nn.Module):
    def __init__(self, num_actions):
        super(cnn_net, self).__init__()
        self.cnn_layer = deepmind()
        self.critic = nn.Linear(512, 1)
        self.actor = nn.Linear(512, num_actions)

        # init the linear layer..
        nn.init.orthogonal_(self.critic.weight.data)
        nn.init.constant_(self.critic.bias.data, 0)
        # init the policy layer...
        nn.init.orthogonal_(self.actor.weight.data, gain=0.01)
        nn.init.constant_(self.actor.bias.data, 0)

    def forward(self, inputs):
        x = self.cnn_layer(inputs / 255.0)
        value = self.critic(x)
        pi = F.softmax(self.actor(x), dim=1)
        return value, pi
def get_tensors(obs):
    return torch.tensor(np.transpose(obs, (0, 3, 1, 2)), dtype=torch.float32)


'''
#To Train a Model
if __name__ == '__main__': 
    # get the arguments
    args = get_args()
    # create environments
    envs = SubprocVecEnv([(lambda _i=i: create_single_football_env(args)) for i in range(args.num_workers)], context=None)
    # create networks
    network = cnn_net(envs.action_space.n)
    # create the ppo agent
    ppo_trainer = ppo_agent(envs, args, network)
    ppo_trainer.learn()
    # close the environments
    envs.close()
'''



#To Test
# get the tensors


if __name__ == '__main__':
    args = get_args()
    model_path = './GFPPOmodel.pt'
    env = football_env.create_environment(\
            env_name=args.env_name, stacked=True,
             render=True)
    network = cnn_net(env.action_space.n)
    network.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    # start to do the test
    obs = env.reset()
    for _ in range(10000):
        obs_tensor = get_tensors(np.expand_dims(obs, 0))
        with torch.no_grad():
            _, pi = network(obs_tensor)
        actions = torch.argmax(pi, dim=1).item()
        obs, reward, done, _ = env.step(actions)
        if done:
            obs = env.reset()
    env.close()


