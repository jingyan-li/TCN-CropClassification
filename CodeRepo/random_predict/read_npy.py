import numpy as np

cof_path = "log/random_prediction/cof_mat.npy"
cof_mat = np.load(cof_path)
overall_accuracy = np.sum(np.diagonal(cof_mat)) / np.sum(cof_mat)
print(f"Overall accuracy: {overall_accuracy}")