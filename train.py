import random
import numpy as np
from model import dqnn_model
import sys
from datapre import datapre
from collections import deque
from target_model import target_model
sys.path.append("game/")
import wrapped_flappy_bird
FINAL_EPSILON = 0
INIT_EPSILON = 1
EPSILON_RATE = 0.99999
REPLAY_MEMORY_SIZE = 50000
BATCH_SIZE = 64
game_state = wrapped_flappy_bird.GameState()
GAMMA = 0.9


def action_bol_flappy(bol_flappy):
    action_bol = np.zeros([2])
    action_bol[bol_flappy] = 1
    return action_bol


def reset():
    image_data, reward, terminal = game_state.frame_step(
        action_bol_flappy(0))
    image_data = datapre(image_data)
    state = np.stack((image_data, image_data, image_data, image_data), axis=3)
    return state


def epsilon_greedy(model, epsilon, state):
    if random.random() <= epsilon:
        if random.random() < 0.95:
            action_bol_one = 0
        else:
            action_bol_one = 1
    else:
        Q_list = model.predict(state)
        action_bol_one = np.argmax(Q_list)
    action_t = action_bol_flappy(action_bol_one)
    return action_t, action_bol_one


def step_t_2_t1(action_t, state_t):
    image_data, reward_t, terminal = game_state.frame_step(action_t)
    image_data = datapre(image_data)
    image_data = np.reshape(image_data, (1, 64, 64, 1))
    state_t1 = np.append(image_data, state_t[:, :, :, :3], axis=3)
    return state_t1, reward_t, terminal


def train():
    t = 0
    epsilon = INIT_EPSILON
    model = dqnn_model()
    model.save_weights("train_" + str(0) + ".h5", overwrite=True)
    target_model_pre = target_model()
    state_t = reset()
    replay_memeory = deque()
    for step in range(9999999999):
        action_t, action_bol_one = epsilon_greedy(model, epsilon, state_t)
        if epsilon > FINAL_EPSILON:
            epsilon = epsilon * EPSILON_RATE
        state_t1, reward_t, terminal = step_t_2_t1(action_t, state_t)
        if len(replay_memeory) < REPLAY_MEMORY_SIZE:
            replay_memeory.append(
                (state_t, action_bol_one, reward_t, state_t1, terminal))
        else:
            replay_memeory.popleft()
            replay_memeory.append(
                (state_t, action_bol_one, reward_t, state_t1, terminal))
        if t > BATCH_SIZE + 1:

            print(len(replay_memeory))
            replay_memeory_sample = random.sample(replay_memeory, BATCH_SIZE)
            state_sample = [batch[0] for batch in replay_memeory_sample]
            action_sample = [batch[1] for batch in replay_memeory_sample]
            reward_sample = [batch[2] for batch in replay_memeory_sample]
            state1_sample = [batch[3] for batch in replay_memeory_sample]
            state_sample_in = np.zeros((BATCH_SIZE, 64, 64, 4))
            target_q_values = np.zeros((BATCH_SIZE, 2))
            for i in range(BATCH_SIZE):
                state_sample_in[i:i + 1] = state_sample[i]
                target_q_values[i] = model.predict(state_sample[i])
                Q_list_t1 = target_model_pre.predict(state1_sample[i])
                if replay_memeory_sample[i][4]:
                    target_q_values[i, action_sample[i]] = reward_sample[i]
                else:
                    target_q_values[i, action_sample[i]
                                    ] = reward_t + GAMMA * np.max(Q_list_t1)
            loss = model.train_on_batch(state_sample_in, target_q_values)
        state_t = state_t1
        t = t + 1
        if t % 1000 == 0:
            target_model_pre.load_weights("train_" + str(t - 1000) + ".h5")
            model.save_weights("train_" + str(t) + ".h5", overwrite=True)
            print("第" + str(t) + "步: " + " loss =" + str(loss) +
                  " epsilon=" + str(epsilon) + " MaxQ=" + str(np.max(Q_list_t1)))


if __name__ == '__main__':
    train()
