import numpy as np


def get_confmat_metrics(confusion_matrix):
    precision = np.diagonal(confusion_matrix) / np.sum(confusion_matrix, axis=0)  # TP/P
    recall = np.diagonal(confusion_matrix) / np.sum(confusion_matrix, axis=1)  # TP/T
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def get_label_weights(weighting_method, label_counts, beta=0.99):
    '''
    Calculate label weights for loss function
    :param weighting_method: 'ivs' - inverse of no. of samples; 'ivs-sqrt' - inverse of square root of no. of samples; 'ens' - effective number of samples
    :param label_counts: python list of size [no_of_classes]
    :param beta: parameter for ens
    :return:
    '''
    if weighting_method == "ivs":
        return get_weights_inverse_num_of_samples(label_counts, power=1.)
    elif weighting_method == "ivs_sqrt":
        return get_weights_inverse_num_of_samples(label_counts, power=0.5)
    elif weighting_method == "ens":
        return get_weights_effective_num_of_samples(label_counts, beta=beta)
    else:
        assert f"Weighting method {weighting_method} is not correct. It should be either 'ivs','ivs_sqrt','ens'."
        return None


def get_weights_inverse_num_of_samples(label_counts, power=1.):
    no_of_classes = len(label_counts)
    weights_for_samples = 1.0/np.power(np.array(label_counts), power)
    weights_for_samples = weights_for_samples/ np.sum(weights_for_samples)*no_of_classes
    return weights_for_samples


def get_weights_effective_num_of_samples(label_counts, beta):
    no_of_classes = len(label_counts)
    label_counts = np.array(label_counts)
    effective_num = 1.0 - np.power(beta, label_counts)
    weights_for_samples = (1. - beta) / effective_num
    weights_for_samples = weights_for_samples / np.sum(weights_for_samples) * no_of_classes
    return weights_for_samples