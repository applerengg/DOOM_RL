######################################
# Alp Eren Gençoğlu, Ziya Ata Yazıcı #
# June 2024                          #
######################################

import itertools as it
import os
import random
from collections import deque
from time import sleep, time

import numpy as np
import skimage.transform
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

import vizdoom as vzd

import cv2

from scipy.io import wavfile

#deterministic random seed
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

best_test_score = float("-inf")

# Q-learning settings
learning_rate = 0.00025
discount_factor = 0.99
train_epochs = 50
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

config = "deadly_corridor"

model_savefile = "Projects/ViZDoom/BLG521E/attn_audio/" + config + "/" + config + "_baseline_baseModel.pth"
save_model = True
load_model = False
skip_learning = False


# Configuration file path
config_file_path = os.path.join(vzd.scenarios_path, config + ".cfg")
# config_file_path = os.path.join(vzd.scenarios_path, "rocket_basic.cfg")
# config_file_path = os.path.join(vzd.scenarios_path, "basic.cfg")

#create folder if not exist
if not os.path.exists("Projects/ViZDoom/BLG521E/attn_audio/" + config):
    os.makedirs("Projects/ViZDoom/BLG521E/attn_audio/" + config)


# Uses GPU if available
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device("cpu")

print("Device: ", DEVICE, flush=True)


def preprocess(img):
    """Down samples image to resolution"""
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img


def create_simple_game():
    print("Initializing doom...", flush=True)
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_seed(seed)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)

    AUDIO_BUFFER_ENABLED = True
    game.set_audio_buffer_enabled(AUDIO_BUFFER_ENABLED)
    game.set_audio_sampling_rate(vzd.SamplingRate.SR_22050)
    game.set_audio_buffer_size(4)

    game.set_render_hud(True)
    game.set_render_minimal_hud(False)
    game.add_game_args("+snd_efx 0")
    game.init()
    print("Doom initialized.", flush=True)

    return game


def test(game, agent):
    """Runs a test_episodes_per_epoch episodes and prints the result"""
    print("\nTesting...")

    test_scores = []
    for test_episode in range(test_episodes_per_epoch):
        game.new_episode()
        while not game.is_episode_finished():
            state_all = game.get_state()

            state = preprocess(state_all.screen_buffer)
            state = np.moveaxis(state, -1, 1)
            state = np.squeeze(state)

            state_audio: np.ndarray = state_all.audio_buffer

            best_action_index = agent.get_action(state, state_audio)

            game.make_action(actions[best_action_index], frame_repeat)
        r = game.get_total_reward()
        test_scores.append(r)

    test_scores = np.array(test_scores)
    print(
        "Results: mean: {:.1f} +/- {:.1f},".format(
            test_scores.mean(), test_scores.std()
        ),
        "min: %.1f" % test_scores.min(),
        "max: %.1f" % test_scores.max(), flush=True,
    )

    global best_test_score
    if test_scores.mean() > best_test_score:
        best_test_score = test_scores.mean()
        print("New best test score:", best_test_score, flush=True)
        if save_model:
            print("Saving the network weights to:", model_savefile, flush=True)
            torch.save(agent.q_net, model_savefile)


def run(game, agent, actions, num_epochs, frame_repeat, steps_per_epoch=2000):
    """
    Run num epochs of training episodes.
    Skip frame_repeat number of frames after each action.
    """

    start_time = time()

    for epoch in range(num_epochs):
        mode = "train"
        game.new_episode()
        train_scores = []
        global_step = 0
        print(f"\nEpoch #{epoch + 1}", flush=True)
        video = cv2.VideoWriter("Projects/ViZDoom/BLG521E/attn_audio/" + config + "/" + config + "_Epoch_" + str(epoch) + ".avi",   
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                         8, (640,480))
        video_frames = []

        for _ in range(steps_per_epoch):
            state_all = game.get_state()
            state_original = state_all.screen_buffer

            #convert to rgb
            state_original = cv2.cvtColor(state_original, cv2.COLOR_BGR2RGB)

            video_frames.append(state_original)

            state = preprocess(state_original)
            #channel first
            state = np.moveaxis(state, -1, 1)
            state = np.squeeze(state)

            state_audio: np.ndarray = state_all.audio_buffer

            action = agent.get_action(state, state_audio)

            reward = game.make_action(actions[action], frame_repeat)
            done = game.is_episode_finished()

            if not done:
                next_state_all = game.get_state()
                next_state = preprocess(next_state_all.screen_buffer)
                #move the channel first instead of 1
                next_state = np.moveaxis(next_state, -1, 1)
                next_state = np.squeeze(next_state)

                next_state_audio: np.ndarray = next_state_all.audio_buffer

            else:
                next_state = np.zeros((3, 30, 45)).astype(np.float32)
                next_state_audio = np.zeros(state_audio.shape).astype(np.float32)

            agent.append_memory(state, state_audio, action, reward, next_state, next_state_audio, done)

            if global_step > agent.batch_size:
                agent.train()

            if done:
                train_scores.append(game.get_total_reward())
                game.new_episode()

            global_step += 1

        for frame in video_frames:
            video.write(frame)

        video.release()

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
        print("Total elapsed time: %.2f minutes" % ((time() - start_time) / 60.0), flush=True)

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
            nn.Conv2d(3, 8, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )

        self.attn_gate1 = nn.Sequential(
            nn.Conv2d(8, 1, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )

        self.attn_gate2 = nn.Sequential(
            nn.Conv2d(8, 1, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )

        self.attn_gate3 = nn.Sequential(
            nn.Conv2d(8, 1, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid(),
        )


        self.conv4 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.attn_gate4 = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid(),
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

    def forward(self, x, audio:torch.Tensor, mod="train"):

        audio = audio.permute(0, 2, 1)
        audio = self.audio_module(audio)

        x = self.conv1(x)
        attn_gate = self.attn_gate1(x)
        x = x * attn_gate

        attn_map = x.detach().cpu().numpy().copy()

        x = self.conv2(x)
        attn_gate = self.attn_gate2(x)
        x = x * attn_gate

        x = self.conv3(x)
        attn_gate = self.attn_gate3(x)
        x = x * attn_gate

        x = self.conv4(x)
        attn_gate = self.attn_gate4(x)
        x = x * attn_gate

        x = x.contiguous().view(-1, 192)
        x = torch.cat([x, audio], dim=1)
        x = self.vision_audio_fuse_module(x)

        x1 = x[:, :96]  # input for the net to calculate the state value
        x2 = x[:, 96:]  # relative advantage of actions in the state
        state_value = self.state_fc(x1).reshape(-1, 1)
        advantage_values = self.advantage_fc(x2)
        x = state_value + (
            advantage_values - advantage_values.mean(dim=1).reshape(-1, 1)
        )
        
        if(mod=="train"):
            return x
        return x, attn_map


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
            print("Loading model from: ", model_savefile, flush=True)
            self.q_net = torch.load(model_savefile)
            self.target_net = torch.load(model_savefile)
            self.epsilon = self.epsilon_min

        else:
            print("Initializing new model", flush=True)
            self.q_net = DuelQNet(action_size).to(DEVICE)
            self.target_net = DuelQNet(action_size).to(DEVICE)

            #print model size
            print("Model Size (MB): ", sum(p.numel() for p in self.q_net.parameters())*4/1024/1024, "MB", flush=True)
            print("Model Parameters (M)", sum(p.numel() for p in self.q_net.parameters())/1000000, "M", flush=True)

        self.opt = optim.SGD(self.q_net.parameters(), lr=self.lr)

    def get_action(self, state, state_audio, mod = "train"):
        if (np.random.uniform() < self.epsilon) & (mod=="train"):
            return random.choice(range(self.action_size))
        else:
            state = np.expand_dims(state, axis=0)
            state = torch.from_numpy(state).float().to(DEVICE)

            state_audio = np.expand_dims(state_audio, axis=0)
            state_audio = torch.from_numpy(state_audio).float().to(DEVICE)

            if(mod=="train"):
                return torch.argmax(self.q_net(state, state_audio)).item()
            else:
                logits, attn = self.q_net(state, state_audio, mod)
                action = torch.argmax(logits).item()
                return action, attn

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

        print("======================================", flush=True)
        print("Training finished. It's time to watch!", flush=True)

    # Reinitialize the game with window visible
    game.close()
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.ASYNC_PLAYER)
    game.init()

    #load the best model
    agent.q_net = torch.load(model_savefile)
    agent.q_net.eval()

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    game.set_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    total_reward = 0
    for ep in range(episodes_to_watch):
        game.new_episode()

        if(skip_learning): 
            path = "Projects/ViZDoom/BLG521E/attn_audio/" + config + "/" + config + "_Final_E"+str(ep)+".avi"
            path_attn = "Projects/ViZDoom/BLG521E/attn_audio/" + config + "/" + config + "_Attn_Final_E" + str(ep) + ".avi"
            path_audio = "Projects/ViZDoom/BLG521E/attn_audio/" + config + "/" + config + "_Audio_Final_E" + str(ep) + ".wav"
        else: 
            path = "Projects/ViZDoom/BLG521E/attn_audio/" + config + "/" + config + "_Final_E"+str(ep)+".avi"
            path_attn = "Projects/ViZDoom/BLG521E/attn_audio/" + config + "/" + config + "_Attn_Final_E"+str(ep)+".avi"
            path_audio = "Projects/ViZDoom/BLG521E/attn_audio/" + config + "/" + config + "_Audio_Final_E"+str(ep)+".wav"

        video = cv2.VideoWriter(path,   
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                         16, (640,480))
        
        video_attn = cv2.VideoWriter(path_attn,   
                    cv2.VideoWriter_fourcc(*'MJPG'), 
                    8, (640,480))

        video_frames_test = []
        attn_frames_test = []
        audio_slices = []

        while not game.is_episode_finished():
            state_all = game.get_state()
            state_original = state_all.screen_buffer

            #convert to rgb
            state_original = cv2.cvtColor(state_original, cv2.COLOR_BGR2RGB)

            video_frames_test.append(state_original)

            state = preprocess(game.get_state().screen_buffer)
            state = np.moveaxis(state, -1, 1)
            state = np.squeeze(state)

            state_audio: np.ndarray = state_all.audio_buffer

            audio_slices.append(state_audio)

            best_action_index, attn = agent.get_action(state, state_audio, "test")

            #make attn maps same size with the frame by interpolating it
            attn = cv2.resize(attn[0, 0], (640, 480))
            attn = cv2.applyColorMap((attn * 255).astype(np.uint8), cv2.COLORMAP_JET)
            #superimpose the frame with the attn map
            attn = cv2.addWeighted(state_original, 0.5, attn, 0.5, 0)
            attn_frames_test.append(attn)

            game.set_action(actions[best_action_index])
            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            for _ in range(1):
                game.advance_action()
            

        for frame in video_frames_test:
            video.write(frame)

        for frame in attn_frames_test:
            video_attn.write(frame)

        wavfile.write(path_audio, 22050, np.concatenate(audio_slices, axis=0))

        video.release()
        video_attn.release()

        # Sleep between episodes
        sleep(1.0)
        score = game.get_total_reward()
        total_reward += score
        print("Total score: ", score, flush=True)

    #mean
    print("Mean score: ", total_reward/episodes_to_watch, flush=True)
