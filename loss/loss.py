import numpy as np
from abc import ABC,abstractmethod

class BaseFunction(ABC):
    @abstractmethod
    def __call__(self, X):
        pass

    @abstractmethod
    def grad(self, X):
        pass


class ReLU(BaseFunction):
    def __call__(self, X):
        return np.maximum(np.zeros_like(X), X)

    def grad(self, X):
        return np.where(X >= 0, 1, 0)


class LeakyReLU(BaseFunction):
  def __init__(self, alpha=0.01):
    self.alpha = alpha
  def __call__(self, X):
    return np.where(X >= 0, X, self.alpha * X)
  def grad(self, X):
    return np.where(X >= 0, 1, self.alpha)

import math

class Softmax(BaseFunction):
    def __call__(self, X):

        if len(X.shape) == 1:
            exp_values = np.exp(X - np.max(X))
            softmax_values = exp_values / np.sum(exp_values)
        else:
            exp_values = np.exp(X - np.max(X, axis=1, keepdims=True))
            softmax_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return softmax_values

    def grad(self, X):
        return 1 # descarte esse gradiente


class CrossEntropy(BaseFunction):
    def __call__(self, Y, Y_pred):
        """
        Argumentos:
        Y: (np.array) labels verdadeiros
        Y_pred: (np.array) labels preditos

        Retorna:
        Saída da Cross Entropy
        """
        
        epsilon=np.float128(1e-8)

        Y = np.array(Y, dtype=np.float64)
        Y_pred = np.array(Y_pred, dtype=np.float64)
        Y_pred = np.clip(Y_pred, epsilon,1-epsilon)

        # Calculate the cross-entropy loss
        loss = -(np.sum(Y * np.log(Y_pred + epsilon)) / len(Y))
        return loss

    def grad(self, Y, Y_pred):
        return Y_pred - Y # gradiente em relação à entrada do Softmax