from abc import ABC, abstractmethod


class BaseOptimizer(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def step(self, grads):
        """
        Argumentos:
        grads: (list) uma lista de tuplas de matrizes (gradiente de pesos, gradiente de bias)
        ambos em formato np.array.

        Retorna:
        """
        pass

class SGDOptimizer(BaseOptimizer):
    def __init__(self, model, lr=1e-3):
        self.model = model
        self.lr = lr

    def step(self, grads: List):
        """
        Argumentos:
        grads: (list) uma lista de tuplas de matrizes (gradiente de pesos, gradiente de bias)
        ambos em formato np.array.

        Retorna:
        """
        for l in range(len(self.model.weights)):
            weight_grad, bias_grad = grads[l]
            self.model.weights[l] = self.model.weights[l] - (self.lr * weight_grad)
            self.model.bias[l] = self.model.bias[l] - (self.lr * bias_grad)



class SGDMomentumOptimizer(BaseOptimizer):

    def __init__(self, model, lr=1e-3, momentum=0.7):
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.velocity = None

    def step(self, grads: List):
        if self.velocity is None:
            self.velocity = [np.zeros_like(w) for w in self.model.weights]

        for l in range(len(self.model.weights)):
            weight_grad, bias_grad = grads[l]

            # Update velocity
            self.velocity[l] = self.momentum * self.velocity[l] + weight_grad

            # Update weights and biases using the learning rate and momentum
            self.model.weights[l] = self.model.weights[l] - (self.lr * self.velocity[l])
            self.model.bias[l] = self.model.bias[l] - (self.lr * bias_grad)