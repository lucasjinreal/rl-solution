# -*- coding: utf-8 -*-
# file: q_policy.py
# author: JinTian
# time: 31/07/2017 6:36 PM
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
This file process current reward to ReplayBuffer which convert the reward to
another consider both current and the last reward.
"""
import numpy as np
from reinforce_learn.replay_buffer import ReplayBuffer


def q_policy_reward(buff, current_reward, current_action):
    """
    this method will using ReplayBuffer, consider the current reward and past
    calculate the new reward, calculate the new reward vector
    :param buff:
    :param current_reward:
    :return:
    """
    assert isinstance(buff, ReplayBuffer.buffer), 'buff must be ReplayBuffer object'
    past_rewards = np.asarray([b[2] for b in buff])
    new_reward = current_reward + 0.9 * (np.argmax([r[current_action] for r in past_rewards])
                                         if len(past_rewards) >= 1 else 0)
    return new_reward
