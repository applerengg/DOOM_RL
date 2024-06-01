#!/usr/bin/env python3

# E. Culurciello, L. Mueller, Z. Boztoprak
# December 2020

import itertools as it
import os
import random
from collections import deque
from time import sleep, time

import numpy as np
import skimage.transform
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

import vizdoom as vzd


# Q-learning settings
learning_rate = 0.00025
discount_factor = 0.99
train_epochs = 5
learning_steps_per_epoch = 2000
replay_memory_size = 10000

# NN learning settings
batch_size = 64

# Training regime
test_episodes_per_epoch = 100

# Other parameters
frame_repeat = 12
resolution = (30, 45)
episodes_to_watch = 100

scenario_name = "basic.cfg"
model_savefile = f"./model_{scenario_name}.pth"
save_model = True
load_model = False # False
skip_learning = False # False

# Configuration file path
# config_file_path = os.path.join(vzd.scenarios_path, "simpler_basic.cfg")
# config_file_path = os.path.join(vzd.scenarios_path, "rocket_basic.cfg")
config_file_path = os.path.join(vzd.scenarios_path, scenario_name)

# Uses GPU if available
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device("cpu")

# SET SEED
torch.manual_seed(0)
np.random.seed(0)


def preprocess(img):
    """Down samples image to resolution"""
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img


def create_simple_game():
    print("Initializing doom...")
    game = vzd.DoomGame()
    game.set_seed(0)

    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)

    AUDIO_BUFFER_ENABLED = True
    game.set_audio_buffer_enabled(AUDIO_BUFFER_ENABLED)
    game.set_audio_sampling_rate(vzd.SamplingRate.SR_22050)
    frameskip = 4
    game.set_audio_buffer_size(frameskip)


    game.init()
    print("Doom initialized.")

    return game


def test(game, agent):
    """Runs a test_episodes_per_epoch episodes and prints the result"""
    print("\nTesting...")
    test_scores = []
    for test_episode in trange(test_episodes_per_epoch, leave=False):
        game.new_episode()
        while not game.is_episode_finished():
            state_all = game.get_state()
            state_screen = preprocess(state_all.screen_buffer)
            state_audio: np.ndarray = state_all.audio_buffer

            best_action_index = agent.get_action(state_screen, state_audio)

            game.make_action(actions[best_action_index], frame_repeat)
        r = game.get_total_reward()
        test_scores.append(r)

    test_scores = np.array(test_scores)
    print(
        "Results: mean: {:.1f} +/- {:.1f},".format(
            test_scores.mean(), test_scores.std()
        ),
        "min: %.1f" % test_scores.min(),
        "max: %.1f" % test_scores.max(),
    )


def run(game, agent, actions, num_epochs, frame_repeat, steps_per_epoch=2000):
    """
    Run num epochs of training episodes.
    Skip frame_repeat number of frames after each action.
    """
    # game.add_game_args('+snd_efx 0')

    start_time = time()

    for epoch in range(num_epochs):
        game.new_episode()
        train_scores = []
        global_step = 0
        print(f"\nEpoch #{epoch + 1}")

        import sys
        # print("#"*30, "DENEME")
        for i in trange(steps_per_epoch, leave=False):
            state_all = game.get_state()
            state_screen = preprocess(state_all.screen_buffer)
            state_audio: np.ndarray = state_all.audio_buffer
        
            # print(type(state_audio), state_audio.shape, state_audio.max())
        
            action = agent.get_action(state_screen, state_audio)
            reward = game.make_action(actions[action], frame_repeat)
            done = game.is_episode_finished()

            if not done:
                next_state_all = game.get_state()
                next_state_audio: np.ndarray = next_state_all.audio_buffer
                next_state = preprocess(next_state_all.screen_buffer)
            else:
                next_state = np.zeros((1, 30, 45)).astype(np.float32)
                next_state_audio = np.zeros(state_audio.shape).astype(np.float32)

            agent.append_memory(state_screen, state_audio, action, reward, next_state, next_state_audio, done)

            if global_step > agent.batch_size:
                agent.train()

            if done:
                train_scores.append(game.get_total_reward())
                game.new_episode()

            global_step += 1

        agent.update_target_net()
        train_scores = np.array(train_scores)

        print(
            "Results: mean: {:.1f} +/- {:.1f},".format(
                train_scores.mean(), train_scores.std()
            ),
            "min: %.1f," % train_scores.min(),
            "max: %.1f," % train_scores.max(),
        )

        test(game, agent)
        if save_model:
            print("Saving the network weights to:", model_savefile)
            torch.save(agent.q_net, model_savefile)
        print("Total elapsed time: %.2f minutes" % ((time() - start_time) / 60.0))

    game.close()
    return agent, game


class DuelQNet(nn.Module):
    """
    This is Duel DQN architecture.
    see https://arxiv.org/abs/1511.06581 for more information.
    """

    def __init__(self, available_actions_count):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.state_fc = nn.Sequential(nn.Linear(96, 64), nn.ReLU(), nn.Linear(64, 1))

        self.advantage_fc = nn.Sequential(
            nn.Linear(96, 64), nn.ReLU(), nn.Linear(64, available_actions_count)
        )

        self.audio_module = nn.Sequential(
            nn.Conv1d(2 , 8 , kernel_size=8, stride=4, bias=True),
            nn.Conv1d(8 , 16, kernel_size=8, stride=4, bias=True),
            nn.Conv1d(16, 16, kernel_size=8, stride=8, bias=True), # torch.Size([batch, 16, 19])
            nn.Flatten(), # torch.Size([batch, 304])
            nn.Linear(304, 192) 
        )

        self.vision_audio_fuse_module = nn.Sequential(
            nn.Linear(384, 192)
        )


    def forward(self, x, audio:torch.Tensor):
        audio = audio.permute(0, 2, 1)
        audio = self.audio_module(audio)
        # TODO audio shape -> 64, 192

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, 192)

        x = torch.cat([x, audio], dim=1)
        # TODO concat audio, x

        x = self.vision_audio_fuse_module(x)

        x1 = x[:, :96]  # input for the net to calculate the state value
        x2 = x[:, 96:]  # relative advantage of actions in the state
        state_value = self.state_fc(x1).reshape(-1, 1)
        advantage_values = self.advantage_fc(x2)
        x = state_value + (
            advantage_values - advantage_values.mean(dim=1).reshape(-1, 1)
        )

        return x


class DQNAgent:
    def __init__(
        self,
        action_size,
        memory_size,
        batch_size,
        discount_factor,
        lr,
        load_model,
        epsilon=1,
        epsilon_decay=0.9996,
        epsilon_min=0.1,
    ):
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.discount = discount_factor
        self.lr = lr
        self.memory = deque(maxlen=memory_size)
        self.criterion = nn.MSELoss()

        if load_model:
            print("Loading model from: ", model_savefile)
            self.q_net = torch.load(model_savefile)
            self.target_net = torch.load(model_savefile)
            self.epsilon = self.epsilon_min

        else:
            print("Initializing new model")
            self.q_net = DuelQNet(action_size).to(DEVICE)
            self.target_net = DuelQNet(action_size).to(DEVICE)

        self.opt = optim.SGD(self.q_net.parameters(), lr=self.lr)

    def get_action(self, state, state_audio):
        if np.random.uniform() < self.epsilon:
            return random.choice(range(self.action_size))
        else:
            state = np.expand_dims(state, axis=0)
            state = torch.from_numpy(state).float().to(DEVICE)

            state_audio = np.expand_dims(state_audio, axis=0)
            state_audio = torch.from_numpy(state_audio).float().to(DEVICE)
            
            action = torch.argmax(self.q_net(state, state_audio)).item()
            return action

    def update_target_net(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def append_memory(self, state, state_audio, action, reward, next_state, next_state_audio, done):
        self.memory.append((state, state_audio, action, reward, next_state, next_state_audio, done))

    def train(self):
        batch = random.sample(self.memory, self.batch_size)
        batch = np.array(batch, dtype=object)

        states = np.stack(batch[:, 0]).astype(float)
        audio  = np.stack(batch[:, 1]).astype(float)

        actions = batch[:, 2].astype(int)
        rewards = batch[:, 3].astype(float)
        next_states = np.stack(batch[:, 4]).astype(float)
        next_states_audio = np.stack(batch[:, 5]).astype(float)
        
        dones = batch[:, 6].astype(bool)

        not_dones = ~dones

        row_idx = np.arange(self.batch_size)  # used for indexing the batch

        # value of the next states with double q learning
        # see https://arxiv.org/abs/1509.06461 for more information on double q learning
        with torch.no_grad():
            next_states = torch.from_numpy(next_states).float().to(DEVICE)
            next_states_audio = torch.from_numpy(next_states_audio).float().to(DEVICE)

            idx = row_idx, np.argmax(self.q_net(next_states, next_states_audio).cpu().data.numpy(), 1)
            next_state_values = self.target_net(next_states, next_states_audio).cpu().data.numpy()[idx]
            next_state_values = next_state_values[not_dones]

        # this defines y = r + discount * max_a q(s', a)
        q_targets = rewards.copy()
        q_targets[not_dones] += self.discount * next_state_values
        q_targets = torch.from_numpy(q_targets).float().to(DEVICE)

        # this selects only the q values of the actions taken
        idx = row_idx, actions
        states = torch.from_numpy(states).float().to(DEVICE)
        audio = torch.from_numpy(audio).float().to(DEVICE)
        action_values = self.q_net(states, audio)[idx].float().to(DEVICE)

        self.opt.zero_grad()
        td_error = self.criterion(q_targets, action_values)
        td_error.backward()
        self.opt.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min


if __name__ == "__main__":
    # Initialize game and actions
    game = create_simple_game()
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]

    # Initialize our agent with the set parameters
    agent = DQNAgent(
        len(actions),
        lr=learning_rate,
        batch_size=batch_size,
        memory_size=replay_memory_size,
        discount_factor=discount_factor,
        load_model=load_model,
    )

    # Run the training for the set number of epochs
    if not skip_learning:
        agent, game = run(
            game,
            agent,
            actions,
            num_epochs=train_epochs,
            frame_repeat=frame_repeat,
            steps_per_epoch=learning_steps_per_epoch,
        )

        print("======================================")
        print("Training finished. It's time to watch!")

    # Reinitialize the game with window visible
    game.close()
    game.set_window_visible(True)
    game.set_mode(vzd.Mode.ASYNC_PLAYER)
    game.init()

    for _ in range(episodes_to_watch):
        game.new_episode()
        while not game.is_episode_finished():
            state_all = game.get_state()
            state_screen = preprocess(state_all.screen_buffer)
            state_audio: np.ndarray = state_all.audio_buffer
            
            best_action_index = agent.get_action(state_screen, state_audio)

            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            game.set_action(actions[best_action_index])
            for _ in range(frame_repeat):
                game.advance_action()

        # Sleep between episodes
        sleep(1.0)
        score = game.get_total_reward()
        print("Total score: ", score)

