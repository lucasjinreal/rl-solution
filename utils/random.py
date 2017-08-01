# -*- coding: utf-8 -*-
# file: random.py
# author: JinTian
# time: 24/07/2017 3:17 PM
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
Personal use utils library.
"""
import numpy as np


def gen_indices(a, size):
    assert isinstance(size, int), 'size must be int'
    assert isinstance(a, int), 'a must be int'
    assert a >= size, 'a must be more than size'
    indices = [np.random.randint(0, a) for _ in range(size)]
    return indices


def random_choose_n(a, size=1):
    """
        a can be a list, or numpy NDArray
    :param a:
    :param size:
    :return:
    """
    if isinstance(a, list):
        indices = gen_indices(len(a), size)
        return np.array(a)[indices]
    elif isinstance(a, np.ndarray):
        s = a.shape
        indices = gen_indices(s[0], size)
        return a[indices]
    else:
        raise ValueError('{} not support'.format(type(a).__name__))
