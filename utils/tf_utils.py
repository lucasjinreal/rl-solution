"""
utils using in TensorFlow

Currently Support Those:

* print_tensor(): print out tensor name and shape
*
"""
import os
import tensorflow as tf
from tensorflow.python.framework.ops import Tensor

try:
    import colorama
except ImportError:
    os.system('sudo pip3 install colorama')
    import colorama
from colorama import init, Fore, Back

init()


def print_tensor(tensor, name='tensor'):
    assert isinstance(tensor, Tensor), 'tensor must be a Tensor'
    print(Fore.YELLOW + '[tensor] ' + name + ', shape: ',
          tensor.get_shape(), Fore.RESET)
