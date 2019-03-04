import numpy as np


def make_data(model, dataset):
    pred_dataset = model.predict(dataset, verbose=1)
    return np.concatenate([pred_dataset, dataset], axis=1)