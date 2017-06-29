# -*- coding: utf-8 -*-
# file: solve_cart_pole.py
# author: JinTian
# time: 24/06/2017 11:26 AM
# Copyright 2017 JinTian. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
import keras.backend as K
from keras.layers import Dense, Activation, Dropout, Conv2D, LSTM
from keras.models import Sequential
import keras
from keras.callbacks import EarlyStopping
import gym
import tensorflow as tf
import numpy as np
import json
import os


def build_model(n_observations):
    # build model for cart-pole system
    print('start building model..')
    model = Sequential()
    model.add(Dense(20, input_dim=n_observations, activation='relu'))
    model.add(Dense(30, activation='tanh'))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    print('building model succeed..')
    return model


def predict_next_action(model, current_state):
    # using model to predict next action depends on current state
    assert isinstance(model, Sequential), 'model must be keras sequential model.'
    current_state = np.expand_dims(current_state, axis=0)
    rewards_vec = model.predict(current_state)
    rewards_vec = np.squeeze(rewards_vec, axis=0)
    return rewards_vec


def train_on_memory(model, memory_container, episode):
    # container contains [(observation0, rewards_vec0), (observation1, rewards_vec1),...]
    # x contains model fits X which is the states, y contains model fits y which is the rewards_vec in every state
    x = np.array(memory_container['observations'])
    y = np.array(memory_container['rewards'])

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, verbose=1, patience=2)
    model.fit(x, y, epochs=3000, validation_split=0.3, callbacks=[early_stopping])
    return model


def policy_gradient():
    env = gym.make('CartPole-v0')
    env.seed(1)
    env = env.unwrapped
    print('==== env information:')
    print(env.action_space.n)
    print(env.observation_space.shape[0])
    print(env.observation_space)

    n_observations = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # this memory container contains all memory
    memory_container = dict()
    model = build_model(n_observations)
    if os.path.exists('cart_pole_500.h5'):
        model.load_weights('cart_pole_500.h5')
        print('weights loaded.')

    episodes = 2000
    print('start episodes...')
    for i in range(episodes):
        env.reset()
        observation, _, _, _ = env.step(env.action_space.sample())

        # init memory observations container and rewards container
        observations = []
        rewards = []

        c = 0
        while True:
            # there is prob to take action from predict and random action to explore
            if np.random.random() < 0.9:
                rewards_vec = predict_next_action(model, observation)
                action = np.argmax(rewards_vec)
            else:
                action = env.action_space.sample()
                rewards_vec = np.zeros_like([1, n_actions])
            print(f'Episode: {i}, Attempt: {c}, action: {action}, rewards_vec: {rewards_vec}')

            # action is single int, 0 or 1 or 2
            observation_, reward, done, info = env.step(action)

            # update execute that action got reward, and look back for the last
            rewards_vec[action] = reward + 0.9 * (np.argmax([r[action] for r in rewards]) if len(rewards) >= 1 else 0)
            rewards.append(rewards_vec)
            observations.append(observation)

            # under observation state, doing action, get reward, store it into memory
            memory_container['observations'] = observations
            memory_container['rewards'] = rewards

            if done:
                # indicates that above actions are all available, if dead then start over again
                print('-' * 70)
                print(f'Episode {i} is done, start train this episode.')
                print(f'collected {len(observations)} samples. {len(observations) == len(rewards)}')
                model = train_on_memory(model, memory_container, i)
                print('-' * 70)
                if i % 500 == 0 and i != 0:
                    model.save_weights(f'cart_pole_{i}.h5', overwrite=True)
                    with open('cart_pole.json', 'w') as f:
                        json.dump(model.to_json(), f)
                    print('model and weights has been saved.')
                break
            # continue to the next step
            observation = observation_
            c += 1
    print('starting to test agent..')


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)
    policy_gradient()
