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
"""
this file shows how to using ReinforcementLearning play Atari breakout game
"""
import keras.backend as K
from keras.layers import Dense, Activation, Dropout, Conv2D, LSTM, Flatten
from keras.models import Sequential
from keras.callbacks import EarlyStopping
import gym
import tensorflow as tf
import numpy as np
import json
import os
import sys


def build_model(observation_shape, action_dim):
    # build model for cart-pole system
    print('start building model..')
    model = Sequential()
    model.add(Conv2D(input_shape=observation_shape, filters=60, kernel_size=(2, 2), padding='SAME', activation='relu'))
    model.add(Conv2D(filters=20, kernel_size=(2, 2), padding='SAME', activation='relu'))
    model.add(Flatten())
    model.add(Dense(20, activation='relu'))
    model.add(Dense(action_dim, activation='sigmoid'))
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
    if x.shape[0] >= 1:
        # 如果memory container是空就直接跳过吧
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, verbose=1, patience=2)
        model.fit(x, y, epochs=3000, validation_split=0.3, callbacks=[early_stopping])
    return model


def experience_policy(attempts):
    # get the experience predict prob according to the attempts
    # if attempts more, indicates Agent more handle, experience more skill, more prob
    if attempts < 30:
        return 0.2
    elif attempts < 200:
        return 0.3
    elif attempts < 600:
        return 0.4
    elif attempts < 1000:
        return 0.5
    elif attempts < 2000:
        return 0.6
    elif attempts < 2500:
        return 0.65
    elif attempts < 3000:
        return 0.68
    elif attempts < 4000:
        return 0.7
    elif attempts < 5000:
        return 0.74
    else:
        return 0.8


def policy_gradient():
    env = gym.make('Breakout-v0')
    env.seed(1)
    env = env.unwrapped
    print('==== env information:')
    print('action dim: {}'.format(env.action_space.n))
    print('observation shape: {}'.format(env.observation_space.shape))

    observation_shape = env.observation_space.shape
    n_actions = env.action_space.n

    # this memory container contains all memory
    memory_container = dict()
    model = build_model(observation_shape, n_actions)
    if os.path.exists('out_of_box_models/breakout_500.h5'):
        model.load_weights('out_of_box_models/breakout_500.h5')
        print('weights loaded.')

    episodes = 2000
    print('start episodes...')
    for i in range(episodes):

        # every episodes prepare job
        env.reset()
        observation, _, _, _ = env.step(env.action_space.sample())
        # empty observations container and rewards container every time
        observations = []
        rewards = []
        c = 0

        while True:
            # there is a prob to take action from predict and random action to explore
            # model predict prob must increasing with c
            model_predict_prob = experience_policy(c)
            if np.random.random() < model_predict_prob:
                rewards_vec = predict_next_action(model, observation)
                action = np.argmax(rewards_vec)
            else:
                action = env.action_space.sample()
                rewards_vec = np.zeros([1, n_actions])[0]
            print('Episode: {}, Attempt: {}, action: {}, rewards_vec: {}'.format(i, c, action, rewards_vec))

            # action is single int, 0 or 1 or 2 or 3
            # core action !!!!!!
            observation_, reward, done, info = env.step(action)

            if done:
                # indicates that above actions are all available, if dead then start over again
                print('-' * 70)
                print('Episode {} is done, start train this episode.'.format(i))
                print('collected {} samples. {}'.format(len(observations) ,len(observations) == len(rewards)))
                model = train_on_memory(model, memory_container, i)
                print('-' * 70)
                if i % 500 == 0 and i != 0:
                    model.save_weights('cart_pole_{}.h5'.format(i), overwrite=True)
                    with open('cart_pole.json', 'w') as f:
                        json.dump(model.to_json(), f)
                    print('model and weights has been saved.')
                break
            else:
                # update execute that action got reward, and look back for the last
                rewards_vec[action] = reward + 0.9\
                                               * (np.argmax([r[action] for r in rewards]) if len(rewards) >= 1 else 0)
                rewards.append(rewards_vec)
                observations.append(observation)

                # under observation state, doing action, get reward, store it into memory
                memory_container['observations'] = observations
                memory_container['rewards'] = rewards
            # continue to the next step
            observation = observation_
            c += 1
    print('starting to test agent..')


def dummy_show():
    env = gym.make('Breakout-v0')
    env.reset()
    env.seed(1)
    for i in range(3000):
        action = env.action_space.sample()
        observation, _, _, _ = env.step(action)
        env.render()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 't':
        dummy_show()
    else:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        K.set_session(sess)
        policy_gradient()

if __name__ == '__main__':
    main()

