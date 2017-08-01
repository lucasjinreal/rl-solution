"""
teach AI play with Ping-Pong Ball
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

from reinforce_learn.model_predict_prob import model_predict_prob
from reinforce_learn.replay_buffer import ReplayBuffer


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


def policy_gradient(is_train=True):
    env = gym.make('Pong-v0')
    env.seed(1)
    env = env.unwrapped
    print('==== env information:')
    print('action dim: {}'.format(env.action_space.n))
    print('observation shape: {}'.format(env.observation_space.shape))

    # rl params
    episodes = 2000
    attempts = 5000
    buffer_size = 5000
    epochs = 500
    batch_size = 32

    observation_shape = env.observation_space.shape
    n_actions = env.action_space.n

    buff = ReplayBuffer(buffer_size=buffer_size)
    model = build_model(observation_shape, n_actions)
    model_save_path = 'out_of_box_models/pong.h5'
    if os.path.exists(model_save_path):
        model.load_weights(model_save_path)
        print('weights loaded.')

    print('start episodes...')
    for i in range(episodes):

        # every episodes prepare job
        env.reset()
        observation, _, _, _ = env.step(env.action_space.sample())

        attempts_count = 0
        while True:
            p = model_predict_prob(attempts=attempts_count)
            if np.random.random() < p:
                observation = np.expand_dims(observation, 0)
                rewards_vec = model.predict(observation)[0]
                action = np.argmax(rewards_vec)
            else:
                action = env.action_space.sample()
                rewards_vec = np.zeros([1, n_actions])[0]

            observation_, reward, done, info = env.step(action)
            print('Episode: {}, Attempt: {}, action: {}, rewards_vec: {}'.format(i, attempts_count, action,
                                                                                 np.around(rewards_vec, 3)))
            # 在这个问题里面，我觉得不能把20分以前的全部拿来训练，应该是reward大于0也就是没有失球的拿来训练
            # 先打印一下每次的reward变化规律, 实时证明每次reward为-1之后，这就得训练了，reward为-1我就看错你死掉了，因为你失球了
            print('reward: ', reward)
            env.render()
            observation = observation_

            # ====== Q Learning Policy Update Rewards ============
            past_rewards = np.asarray([b[2] for b in buff.buffer])
            new_reward = reward + 0.9 * (np.argmax([r[action] for r in past_rewards])
                                         if len(past_rewards) >= 1 else 0)
            rewards_vec[action] = new_reward
            buff.add_simple(observation, action, rewards_vec, done)
            attempts_count += 1
            if reward == -1:
                print('==== Episode {} is done, start train on this episode:'.format(i))

                states = np.asarray([e[0] for e in buff.buffer])
                rewards = np.asarray([e[2] for e in buff.buffer])

                early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, verbose=1, patience=2)
                try:
                    model.fit(states, rewards, epochs=epochs, validation_split=0.3, callbacks=[early_stopping])
                    model.save_weights(model_save_path)
                except KeyboardInterrupt:
                    model.save_weights(model_save_path)
                # after train, erase memory to the next episode
                buff.erase()
                break

    print('agent train finished..')


def dummy_show():
    env = gym.make('Pong-v0')
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
