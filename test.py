# -*- coding: utf-8 -*-
# file: test.py
# author: JinTian
# time: 29/06/2017 10:34 AM
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
import numpy as np
import matplotlib.pyplot as plt


def fx(x):
    return 1 - 1/np.exp(x)


def fx2(x):
    return np.exp(x/200)


def test():
    a = np.zeros([1, 4])[0]
    print(a)



if __name__ == '__main__':
    test()