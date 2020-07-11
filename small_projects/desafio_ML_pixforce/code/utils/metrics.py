####################################################################################################
# Pixforce                                                                                         #
# Desenvolvido por: Rafael Padilla (eng.rafaelpadilla@gmail.com)                                   #
# Última atualização: 11 julho 2020                                                                #
####################################################################################################

import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)


def calculate_accuracy(TP, FP, FN, TN):
    return (TP+TN)/(TP+FP+FN+TN)


def calculate_precision(TP, FP, FN, TN):
    return TP / (TP + FP)


def calculate_recall(TP, FP, FN, TN):
    return TP / (TP+FN)


def calculate_f1_score(TP, FP, FN, TN):
    precision = calculate_precision(TP, FP, FN, TN)
    recall = calculate_recall(TP, FP, FN, TN)
    return 2 * (precision * recall) / (precision + recall)


def calculate_metrics(y_pred, y_gt):
    '''Calcula as seguintes métricas dado uma lista com classes ground-truth e predições obtidas
    por um classificador:
        TP: quantidade de verdadeiros positivos.
        FP: quantidade de falsos positivos.
        TN: quantidade de verdadeiros negativos.
        FN: quantidade de falsos negativos.
        acurácia: (TP+TN) / (TP+FP+FN+TN).
        f1: f1 score.
        precisão: porcentagem de predições positivas corretas.
        revocação: porcentagem de TP detectados considerando todos os ground-truths.

        y_pred (list) : lista com as classes retornadas pelo classificador (Ex: [0, 0, 1, 1, 0, 1]).
        y_gt (list)   : lista com as classes ground-truth (Ex: [0, 0, 0, 1, 1, 0]).
precision
        metrics (dict) : dicionário contendo as métricas.
    '''
    # Transfere para cpu para economizar memória
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if isinstance(y_gt, torch.Tensor):
        y_gt = y_gt.cpu().numpy()
    # True Positive, False Positive, False Negative, True Negative
    TP = sum([1 if pred+gt == 2 else 0 for pred, gt in zip(y_pred, y_gt)])
    FP = sum([1 if pred-gt == 1 else 0 for pred, gt in zip(y_pred, y_gt)])
    FN = sum([1 if pred-gt == -1 else 0 for pred, gt in zip(y_pred, y_gt)])
    TN = sum([1 if pred+gt == 0 else 0 for pred, gt in zip(y_pred, y_gt)])
    # Precision: TP / (TP + FP)
    if TP + FP == 0:
        precision = 0.
    else:
        precision = calculate_precision(TP, FP, FN, TN)
    assert precision_score(y_gt, y_pred) == precision
    # Recall: TP / (TP+FN)
    if TP+FN == 0:
        recall = 0.
    else:
        recall = calculate_recall(TP, FP, FN, TN)
    assert recall_score(y_gt, y_pred) == recall
    # Accuracy: (TP+TN) / (TP+FP+FN+TN)
    if (TP+FP+FN+TN) == 0:
        accuracy = 0.
    else:
        accuracy = calculate_accuracy(TP, FP, FN, TN)
    assert accuracy_score(y_gt, y_pred) == accuracy
    # F1 score
    if precision + recall == 0:
        f1 = 0.
    else:
        f1 = calculate_f1_score(TP, FP, FN, TN)
    assert f1_score(y_gt, y_pred) == f1
    return {'TP': TP,
            'FP': FP,
            'FN': FN,
            'TN': TN,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1}
