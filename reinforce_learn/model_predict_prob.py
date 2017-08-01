import numpy as np


def model_predict_prob(attempts):
    """
    generate model predict probability, attempts more, model predict more
    :param attempts:
    :return:
    """
    if attempts < 30:
        return 0.5
    elif attempts < 200:
        return 0.6
    elif attempts < 600:
        return 0.65
    elif attempts < 1000:
        return 0.69
    elif attempts < 2000:
        return 0.7
    elif attempts < 2500:
        return 0.75
    elif attempts < 3000:
        return 0.78
    elif attempts < 4000:
        return 0.8
    elif attempts < 5000:
        return 0.84
    else:
        return 0.9
