from typing import *
import os
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import njit
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch.utils.tensorboard import SummaryWriter
from memory import ReplayBuffer, PrioritizedReplayBuffer
from network import Network 


class DQNAgent:
    def __init__(
        self, 
        # General parameters
        log_dir: str,
        ckpt_dir: str,
        load_ckpt: bool,
        writer: SummaryWriter,
        # Environment parameters
        max_days: int = 200,
        treatment_days: int = 1,
        reward_scaler: float = 1e+8,
        # Training parameters
        memory_size: int = int(1e6),
        batch_size: int = 2048,
        lr: float = 2e-4,
        l2_reg: float = 0.,
        grad_clip: float = 1000.,
        target_update: int = 3000,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.05,
        epsilon_decay: float = 1 / 200,
        decay_option: str = 'logistic',
        discount_factor: float = 0.99,
        n_train: int = 1,
        # Network parameters
        hidden_dim: int = 1024,
        # PER parameters
        per: bool = True,
        alpha: float = 0.2,
        beta: float = 0.6,
        beta_increment_per_sampling: float = 0.000005,
        prior_eps: float = 1e-6,
        # Double DQN
        double_dqn: bool = False,
    ):
        # Store parameters
        self.max_days = max_days
        self.treatment_days = treatment_days
        self.reward_scaler = reward_scaler

        # Initial states for different scenarios
        UNHEALTHY_STEADY_INIT_STATE = np.log10(np.array([163573, 5, 11945, 46, 63919, 24], dtype=np.float32))
        HIGH_T_LOW_V_INIT_STATE = np.log10(np.array([1.0e+6, 3198, 1.0e-4, 1.0e-4, 1, 10], dtype=np.float32))
        HIGH_T_HIGH_V_INIT_STATE = np.log10(np.array([1.0e+6, 3198, 1.0e-4, 1.0e-4, 1000000, 10], dtype=np.float32))
        LOW_T_HIGH_V_INIT_STATE = np.log10(np.array([1000, 10, 10000, 100, 1000000, 10], dtype=np.float32))

        # Create environments
        self.envs = {
            'train': make_env(UNHEALTHY_STEADY_INIT_STATE, max_days, treatment_days, reward_scaler),
            'HTLV': make_env(HIGH_T_LOW_V_INIT_STATE, max_days, treatment_days, reward_scaler),
            'HTHV': make_env(HIGH_T_HIGH_V_INIT_STATE, max_days, treatment_days, reward_scaler),
            'LTHV': make_env(LOW_T_HIGH_V_INIT_STATE, max_days, treatment_days, reward_scaler),
        }
        
        # Get environment dimensions
        obs_dim = self.envs['train'].observation_space.shape[0]
        action_dim = self.envs['train'].action_space.n

        # Store parameters
        self.writer = writer
        self.log_dir = log_dir
        self.ckpt_dir = ckpt_dir
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.target_update = target_update
        self.epsilon = max_epsilon
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.decay_option = decay_option
        self.discount_factor = discount_factor
        self.n_train = n_train

        # Device setup
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f'device: {self.device}')
        
        # Setup memory
        self.per = per
        if per:
            self.prior_eps = prior_eps
            self.memory = PrioritizedReplayBuffer(
                obs_dim, memory_size, batch_size, alpha, beta, beta_increment_per_sampling
            )
        else:
            self.memory = ReplayBuffer(obs_dim, memory_size, batch_size)

        # Double DQN flag
        self.double_dqn = double_dqn

        # Setup networks
        dqn_config = dict(
            in_dim=obs_dim,
            nf=hidden_dim,
            out_dim=action_dim,
        )
        self.dqn = Network(**dqn_config).to(self.device)
        self.dqn_target = Network(**dqn_config).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()
        
        # Setup optimizer
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=lr, weight_decay=l2_reg)

        # Training/testing flags
        self.is_test = False
        self.max_cum_reward = -1.
        
        # Records
        self.record = [] 
        self.init_episode = 1

        # Calculate benchmarks
        logging.info('Computing Benchmark... (Wait for few seconds)')
        self.bm_info = {
            'no_drug': {'states': {}, 'actions': {}, 'rewards': {}},
            'full_drug': {'states': {}, 'actions': {}, 'rewards': {}},
        }
        for opt in self.bm_info.keys():
            for name, _env in self.envs.items():
                _states, _actions, _rewards = self._test(_env, opt)
                self.bm_info[opt]['states'][name] = _states 
                self.bm_info[opt]['actions'][name] = _actions
                self.bm_info[opt]['rewards'][name] = _rewards
        logging.info('Done!')

        # Load checkpoint if requested
        if load_ckpt:
            self.load_ckpt()
    
        wandb.init(
            project="hiv-treatment",
            config={
                "lr": lr,
                "batch_size": batch_size,
                "hidden_dim": hidden_dim,
                "max_epsilon": max_epsilon,
                "treatment_days": treatment_days
            }
        )

    def save_ckpt(self, episode: int, path: str) -> None:
        if self.per:
            _memory = _gather_per_buffer_attr(self.memory)
        else:
            _memory = _gather_replay_buffer_attr(self.memory)
        ckpt = dict(
            episode=episode,
            dqn=self.dqn.state_dict(),
            dqn_target=self.dqn_target.state_dict(),
            optim=self.optimizer.state_dict(),
            memory=_memory,
        )
        torch.save(ckpt, path)

    def load_ckpt(self) -> None:
        ckpt = torch.load(os.path.join(self.ckpt_dir, 'ckpt.pt'))
        self.init_episode = ckpt['episode'] + 1
        self.dqn.load_state_dict(ckpt['dqn'])
        self.dqn_target.load_state_dict(ckpt['dqn_target'])
        self.optimizer.load_state_dict(ckpt['optim'])
        for key, value in ckpt['memory'].items():
            if key not in ['sum_tree', 'min_tree']:
                setattr(self.memory, key, value)
            else:
                tree = getattr(self.memory, key)
                setattr(tree, 'capacity', value['capacity'])
                setattr(tree, 'tree', value['tree'])
        logging.info(f'Success: Checkpoint loaded (start from Episode {self.init_episode})!')

    def select_action(self, state: np.ndarray) -> int:
        '''Select an action from the input state using epsilon-greedy policy.'''
        if self.epsilon > np.random.random() and not self.is_test:
            selected_action = self.envs['train'].action_space.sample()
        else:
            selected_action = self.dqn(torch.FloatTensor(state).to(self.device)).argmax()
            selected_action = selected_action.detach().cpu().numpy()
        return selected_action

    def step(self, env: gym.Env, action: int) -> Tuple[np.ndarray, np.float64, bool, Optional[np.ndarray]]:
        '''Take an action and return the response of the env.'''
        next_state, reward, done, _, info = env.step(action)
        return next_state, reward, done, info.get('intermediate_sol', None)

    def update_model(self) -> torch.Tensor:
        '''Update the model by gradient descent.'''
        if self.per:
            # PER needs beta to calculate weights
            samples = self.memory.sample_batch()
            weights = torch.FloatTensor(samples["weights"].reshape(-1, 1)).to(self.device)
            indices = samples["indices"]
        else:
            # Vanilla DQN sampling
            samples = self.memory.sample_batch()
        
        # Calculate loss
        elementwise_loss = self._compute_dqn_loss(samples)
        if self.per:
            loss = torch.mean(elementwise_loss * weights)
        else:
            loss = torch.mean(elementwise_loss)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.dqn.parameters(), self.grad_clip)
        self.optimizer.step()
        
        if self.per:
            # PER: update priorities
            loss_for_prior = elementwise_loss.squeeze().detach().cpu().numpy()
            new_priorities = loss_for_prior + self.prior_eps
            self.memory.update_priorities(indices, new_priorities)

        return loss.item()
        
    def train(self, max_episodes: int, log_freq: int, test_freq: int, save_freq: int, img_dir: str) -> None:
        '''Train the agent.'''
        self.is_test = False
        max_steps = self.envs['train'].max_episode_steps
        update_cnt = 0
        start = datetime.now()

        for episode in range(self.init_episode, max_episodes+1):
            state = self.envs['train'].reset()[0]
            losses = []

            # Episode loop
            for _ in range(max_steps):
                action = self.select_action(state)
                next_state, reward, done, _ = self.step(self.envs['train'], action)
                transition = [state, action, reward, next_state, done]
                self.memory.store(*transition)
                state = next_state

                # Training when enough samples
                if len(self.memory) >= self.batch_size:
                    for _ in range(self.n_train):
                        loss = self.update_model()
                    losses.append(loss)
                    self.writer.add_scalar('loss', loss, update_cnt)
                    update_cnt += 1

                    # Epsilon decay
                    if self.decay_option == 'linear':
                        self.epsilon = max(
                            self.min_epsilon, self.epsilon - (
                                self.max_epsilon - self.min_epsilon
                            ) * self.epsilon_decay
                        )
                    elif self.decay_option == 'logistic':
                        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * \
                            sigmoid(1 / self.epsilon_decay - episode)

                    # Target network update
                    if update_cnt % self.target_update == 0:
                        self._target_hard_update()

                if done:
                    break

            avg_step_train_loss = np.array(losses).sum() * self.batch_size / max_steps

            # Testing
            if test_freq > 0 and episode % test_freq == 0:
                last_treatment_day, max_E, last_E, cum_reward = self.test(episode, img_dir)
                self.record.append({
                    'episode': episode,
                    'last_treatment_day': last_treatment_day,
                    'max_E': max_E,
                    'last_E': last_E,
                    'cum_reward': cum_reward,
                    'train_loss': avg_step_train_loss,
                })
                self._save_record_df()

                wandb.log({
                    "episode": episode,
                    "reward": cum_reward,
                    "loss": avg_step_train_loss,
                    "epsilon": self.epsilon,
                    "max_E": max_E,
                    "last_E": last_E,
                    "last_treatment_day": last_treatment_day
                })

                # Logging
                if log_freq > 0 and episode % log_freq == 0:
                    self._track_results(
                        episode,
                        datetime.now() - start,
                        train_loss=avg_step_train_loss,
                        max_E=max_E,
                        last_E=last_E,
                        cum_reward=cum_reward,
                    )

            # Checkpoint saving
            if save_freq > 0 and episode % save_freq == 0:
                path = os.path.join(self.ckpt_dir, 'ckpt.pt')
                self.save_ckpt(episode, path)
                
        self.envs['train'].close()
                
    def test(self, episode: int, img_dir: str) -> Tuple[int, float, float, float]:
        '''Test the agent and generate visualizations'''
        # Get trajectories for training environment
        _states, _actions, _rewards = self._test(self.envs['train'], 'policy')
        states = {'train': _states}
        actions = {'train': _actions}
        rewards = {'train': _rewards}

        # Calculate cumulative reward
        cum_reward = discounted_sum(rewards['train'], self.discount_factor)
        
        # If performance improved, test on other environments
        if cum_reward > max(1e+0, self.max_cum_reward):
            for name, _env in self.envs.items():
                if name == 'train':
                    continue
                states[name], actions[name], rewards[name] = self._test(_env, 'policy')

            # Generate visualizations
            for env_name in self.envs.keys():
                self._plot_6_states_2_actions(
                    episode, img_dir,
                    states[env_name], actions[env_name],
                    self.bm_info['no_drug']['states'][env_name],
                    self.bm_info['full_drug']['states'][env_name],
                    env_name,
                )

            self._plot_VE_phase_plane(episode, img_dir, states, actions)

        # Cleanup
        for env in self.envs.values():
            env.close()
        self.dqn.train()

        # Calculate metrics
        last_a1_day = get_last_treatment_day(actions['train'][:, 0])
        last_a2_day = get_last_treatment_day(actions['train'][:, 1])
        last_treatment_day = max(last_a1_day, last_a2_day) * (self.max_days // len(actions['train']))
        max_E = 10 ** (states['train'][:, 5].max())
        last_E = 10 ** (states['train'][-1, 5])
        
        if cum_reward > self.max_cum_reward:
            self.max_cum_reward = cum_reward

        return last_treatment_day, max_E, last_E, cum_reward

    def _test(self, env: gym.Env, mode: str = 'policy') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''Test the agent in a specific mode'''
        assert mode in ['policy', 'no_drug', 'full_drug']
        self.is_test = True
        self.dqn.eval()

        max_steps = env.max_episode_steps
        states, actions, rewards = [], [], []

        with torch.no_grad():
            state = env.reset()[0]
            for _ in range(max_steps):
                # Select action based on mode
                if mode == 'policy':
                    action = self.select_action(state)
                elif mode == 'no_drug':
                    action = 0
                elif mode == 'full_drug':
                    action = 3

                next_state, reward, _, intermediate_sol = self.step(env, action)
                _action = np.array(env.action_set[action]).reshape(1, -1)
                _reward = np.array([reward,]) * env.reward_scaler

                # Handle intermediate states
                if intermediate_sol is not None:
                    intermediate_states = intermediate_sol[:6, :].transpose()
                    _state = np.concatenate([state.reshape(1, -1), intermediate_states], axis=0)
                    _action = np.repeat(_action, _state.shape[0], axis=0)
                else:
                    _state = state.reshape(1, -1)

                states.append(_state)
                actions.append(_action)
                rewards.append(_reward)
                state = next_state

        # Concatenate and format results
        states = np.concatenate(states, axis=0, dtype=np.float32)
        actions = np.concatenate(actions, axis=0, dtype=np.float32)
        rewards = np.concatenate(rewards, axis=0, dtype=np.float32).reshape(-1, 1)
        
        self.is_test = False
        return states, actions, rewards

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        '''Compute DQN loss'''
        device = self.device
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # Current Q Value
        curr_q_value = self.dqn(state).gather(1, action)

        # Next Q Value
        if not self.double_dqn:
            next_q_value = self.dqn_target(next_state).max(dim=1, keepdim=True)[0].detach()
        else:
            next_q_value = self.dqn_target(next_state).gather(
                1, self.dqn(next_state).argmax(dim=1, keepdim=True)
            ).detach()

        mask = 1 - done
        target = (reward + self.discount_factor * next_q_value * mask).to(self.device)

        return F.smooth_l1_loss(curr_q_value, target, reduction='none')

    def _target_hard_update(self):
        '''Hard update target network'''
        self.dqn_target.load_state_dict(self.dqn.state_dict())
                
    def _track_results(
        self,
        episodes: int,
        elapsed_time: timedelta,
        train_loss: float,
        max_E: float,
        last_E: float,
        cum_reward: float,
    ):
        '''Log training progress'''
        elapsed_time = str(timedelta(seconds=elapsed_time.seconds))
        logging.info(
            f'Epi {episodes:>4d} | {elapsed_time} | LastE {last_E:8.1f} | CumR {cum_reward:.3e} | '\
            f'Loss (Train) {train_loss:.2e} | Buffer {self.memory.size}'
        )

    def _save_record_df(self):
        '''Save training records to CSV'''
        df = pd.DataFrame(self.record).set_index('episode')
        df.to_csv(os.path.join(self.log_dir, 'records.csv'))

    def _plot_6_states_2_actions(
        self,
        episode: int,
        img_dir: str,
        policy_states: np.ndarray,
        policy_actions: np.ndarray,
        no_drug_states: np.ndarray,
        full_drug_states: np.ndarray,
        env_name: str,
    ) -> None:
        '''Generate state-action visualization'''
        fig = plt.figure(figsize=(14, 18))
        plt.axis('off')

        # Define labels and axes
        state_names = [
            r'$\log_{10}(T_{1}$)', r'$\log_{10}(T_{2})$',
            r'$\log_{10}(T_{1}^{*})$', r'$\log_{10}(T_{2}^{*})$',
            r'$\log_{10}(V)$', r'$\log_{10}(E)$',
        ]
        action_names = [rf'RTI $\epsilon_{1}$', rf'PI $\epsilon_{2}$']
        axis_t = np.arange(policy_states.shape[0]) * self.treatment_days
        label_fontdict = {'size': 13}

        # Plot states
        for i in range(6):
            ax = fig.add_subplot(4, 2, i+1)
            ax.plot(axis_t, policy_states[:, i], label='ours', color='crimson', linewidth=2)
            ax.plot(axis_t, no_drug_states[:, i], label='no drug', color='royalblue', linewidth=2, linestyle='--')
            ax.plot(axis_t, full_drug_states[:, i], label='full drug', color='black', linewidth=2, linestyle='-.')
            ax.set_xlabel('Days', labelpad=0.8, fontdict=label_fontdict)
            ax.set_ylabel(state_names[i], labelpad=0.5, fontdict=label_fontdict)
            if i == 0:
                ax.set_ylim(min(4.8, policy_states[:, i].min() - 0.2), 6)

        # Plot actions
        last_a1_day = get_last_treatment_day(policy_actions[:, 0])
        last_a2_day = get_last_treatment_day(policy_actions[:, 1])
        last_treatment_day = max(last_a1_day, last_a2_day) * (self.max_days // len(policy_actions))

        for i in range(2):
            ax = fig.add_subplot(4, 2, i+7)
            if last_treatment_day < 550:
                if last_a1_day >= last_a2_day:
                    if i == 0:
                        ax.text(last_a1_day * self.treatment_days, -0.07, f'Day {last_a1_day * self.treatment_days}')
                else:
                    if i == 1:
                        ax.text(last_a2_day * self.treatment_days, -0.03, f'Day {last_a2_day * self.treatment_days}')

            _a = np.repeat(policy_actions[:, i], self.treatment_days, axis=0)
            ax.plot(np.arange(policy_states.shape[0] * self.treatment_days), _a, color='forestgreen', linewidth=2)
            
            if i == 0:
                ax.set_ylim(0.7 * (-0.2), 0.7 * 1.2)
                ax.set_yticks([0.0, 0.7])
            else:
                ax.set_ylim(0.3 * (-0.2), 0.3 * 1.2)
                ax.set_yticks([0.0, 0.3])
            ax.set_xlabel('Days', labelpad=0.8, fontdict=label_fontdict)
            ax.set_ylabel(action_names[i], labelpad=0.5, fontdict=label_fontdict)

        fig.savefig(
            os.path.join(img_dir, f'Epi{episode}_{env_name}_{last_treatment_day}.png'),
            bbox_inches='tight',
            pad_inches=0.2,
        )
        plt.close(fig)
        return

    def _plot_VE_phase_plane(
        self,
        episode: int,
        img_dir: str,
        states: Dict[str, np.ndarray],
        actions: Dict[str, np.ndarray],
    ) -> None:
        '''Generate V-E phase plane visualization'''
        label_fontdict = {'size': 13}
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1, projection='3d')

        # Define plotting metadata
        meta_info = {
            'train': dict(color='navy', alpha=0.8, label='train (initial: unhealthy steady state)'),
            'HTLV': dict(color='forestgreen', alpha=0.8, label='test (initial: early infection with one virus)'),
            'HTHV': dict(color='darkorange', alpha=0.8, label=r'test (initial: early infection with $10^6$ virus)'),
            'LTHV': dict(color='indianred', alpha=0.8, label=r'test (initial: small T-cells with $10^6$ virus)'),
        }
        init_labels = ['A', 'B', 'C', 'D']

        # Plot trajectories
        for i, (env_name, kwargs) in enumerate(meta_info.items()):
            _s = states[env_name]
            x = _s[:, 4]  # log(V)
            y = _s[:, 0]  # log(T1)
            z = _s[:, 5]  # log(E)
            
            ax.plot(x, y, z, **kwargs)
            ax.scatter(x[0], y[0], z[0], color='black', marker='o', s=70)
            ax.text(x[0], y[0], z[0] - 0.4, init_labels[i], fontdict=dict(size=13,))

        # Mark endpoint for training trajectory
        ax.scatter(
            states['train'][-1, 4], states['train'][-1, 0], states['train'][-1, 5],
            color='red', marker='*', s=120,
        )
        ax.text(
            states['train'][-1, 4], states['train'][-1, 0], states['train'][-1, 5] + 0.4,
            'End', fontdict=dict(size=14,),
        )

        # Configure view and axes
        ax.view_init(15, 45)
        ax.set_xlabel(r'$\log_{10}(V)$', labelpad=2, fontdict=label_fontdict)
        ax.set_xlim(-0.5, 6.5)
        ax.set_ylabel(r'$\log_{10}(T_{1})$', labelpad=2, fontdict=label_fontdict)
        ax.set_ylim(3, 7)
        ax.set_zlabel(r'$\log_{10}(E)$', labelpad=2, fontdict=label_fontdict)
        ax.set_zlim(0, 6.5)
        ax.legend(loc='upper right')

        # Save plot
        fig.savefig(
            os.path.join(img_dir, f'Epi{episode}_VE.png'),
            bbox_inches='tight',
            pad_inches=0.2,
        )
        plt.close(fig)
        return


def _gather_replay_buffer_attr(memory: Optional[ReplayBuffer]) -> dict:
    '''Gather replay buffer attributes for checkpoint saving'''
    if memory is None:
        return {}
    replay_buffer_keys = [
        'obs_buf', 'next_obs_buf', 'acts_buf', 'rews_buf', 'done_buf',
        'max_size', 'batch_size', 'ptr', 'size',
    ]
    return {key: getattr(memory, key) for key in replay_buffer_keys}


def _gather_per_buffer_attr(memory: Optional[PrioritizedReplayBuffer]) -> dict:
    '''Gather prioritized experience replay buffer attributes for checkpoint saving'''
    if memory is None:
        return {}
    per_buffer_keys = [
        'obs_buf', 'next_obs_buf', 'acts_buf', 'rews_buf', 'done_buf',
        'max_size', 'batch_size', 'ptr', 'size',
        'max_priority', 'tree_ptr', 'alpha',
    ]
    result = {key: getattr(memory, key) for key in per_buffer_keys}
    result['sum_tree'] = dict(
        capacity=memory.sum_tree.capacity,
        tree=memory.sum_tree.tree,
    )
    result['min_tree'] = dict(
        capacity=memory.min_tree.capacity,
        tree=memory.min_tree.tree,
    )
    return result


def sigmoid(x: float) -> float:
    '''Sigmoid function for epsilon decay'''
    return 1 / (1 + np.exp(-x))


@njit(cache=True)
def get_last_treatment_day(action: np.ndarray) -> int:
    '''Find the last treatment day (i.e, nonzero actions) for a given action sequence.'''
    n = len(action)
    for i in range(n-1, -1, -1):
        if action[i] != 0:
            return i + 1
    return 0


@njit(cache=True)
def discounted_sum(rewards: np.ndarray, discount_factor: float = 0.99, n: int = 5) -> float:
    '''Calculate discounted sum of rewards'''
    _sum = 0.
    _factor = 1.
    _cnt = 0
    for r in rewards[:, 0]:
        _sum += r * _factor
        _cnt += 1
        if _cnt % n == 0:
            _factor *= discount_factor 
    return _sum


def make_env(init_state: np.ndarray, max_days: int, treatment_days: int, reward_scaler: float) -> gym.Env:
    '''Create an instance of the HIV environment'''
    from fast_env_py import FastHIVPatient as HIVPatient
    env = HIVPatient()
    
    # Set environment attributes
    env.max_episode_steps = max_days // treatment_days
    env.treatment_days = treatment_days
    env.reward_scaler = reward_scaler
    
    # Set initial state if provided
    if init_state is not None:
        state = 10 ** init_state  # Convert from log scale
        env.T1 = state[0]
        env.T2 = state[1]
        env.T1star = state[2]
        env.T2star = state[3]
        env.V = state[4]
        env.E = state[5]
    
    return env

class ProjectAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dqn = Network(in_dim=6, nf=1024, out_dim=4).to(self.device)

    def load(self):
        checkpoint = torch.load('trained_models/ckpt.pt')
        self.dqn.load_state_dict(checkpoint['dqn'])
        self.dqn.eval()

    def act(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            return self.dqn(state).argmax().item()