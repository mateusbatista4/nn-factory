import numpy as np
from typing import List

from loss.loss import BaseFunction

class Model:
    def __init__(self, layers_dims: List[int],
                 activation_funcs: List[BaseFunction],
                 initialization_method: str = "random"):
        """
        Argumentos:
        layers_dims: (list) lista com o tamanho de cada camada
        activation_funcs: (list) lista com as funções de ativação
        initialization_method: (str) indica como inicializar os parâmetros

        Exemplo:

        # Um modelo de arquitetura com camadas 2 x 1 x 2 e 2 ReLU como funções de ativação
        >>> m = Model([2, 1, 2], [ReLU(), ReLU()])
        """

        assert all([isinstance(d, int) for d in layers_dims]), \
        "É esperado uma lista de int como o parâmetro ``layers_dims"

        assert all([isinstance(a, BaseFunction) for a in activation_funcs]), \
        "É esperado uma lista de BaseFunction como o parâmetro ``activation_funcs´´"

        self.layers_dims = layers_dims
        self.activation_funcs = activation_funcs
        self.weights, self.bias = self.initialize_model(initialization_method)


    def __len__(self):
        return len(self.weights)


    def initialize_model(self, method="random"):
        """
        Argumentos:
        layers_dims: (list)  lista com o tamanho de cada camada
        method: (str) indica como inicializar os parâmetros

        Retorna: uma lista de matrizes (np.array) de pesos e
        uma lista de matrizes (np.array) como biases.
        """

        weights = []
        bias = []
        n_layers = len(self.layers_dims)
        for l in range(0, n_layers-1):
            # o peso w_i,j  conecta o i-th neurônio na camada atual para
            # o j-th neurônio na próxima camada
            W = np.random.randn(self.layers_dims[l], self.layers_dims[l + 1])
            b = np.random.randn(1, self.layers_dims[l + 1])

            # He et al. Inicialização Normal
            if method.lower() == 'he':
                W = W * np.sqrt(2/self.layers_dims[l])
                b = b * np.sqrt(2/self.layers_dims[l])

            elif method.lower() == 'random':
                W = np.random.randn(self.layers_dims[l], self.layers_dims[l + 1])
                b = np.random.randn(1, self.layers_dims[l + 1])

            elif method.lower() == 'xavier':
                W = np.random.randn(self.layers_dims[l], self.layers_dims[l + 1]) * np.sqrt(1/self.layers_dims[l])
                b = np.random.randn(1, self.layers_dims[l + 1]) * np.sqrt(1/self.layers_dims[l])


            weights.append(W)
            bias.append(b)

        return weights, bias


    def forward(self, X):
        """
        Argumentos:
        X: (np.array) dados de entrada

        Retorno:
        Predições para os dados de entrada (np.array)
        """
        activation = X
        self.activations = [X]
        self.Z_list = []
       
        activation = X
        self.activations = [X]
        self.Z_list = []

        for l in range(len(self.layers_dims) - 1):
            W = self.weights[l]
            b = self.bias[l]
            activation_func = self.activation_funcs[l]

            Z = np.dot(activation, W) + b
            activation = activation_func(Z)

            self.Z_list.append(Z)
            self.activations.append(activation)

        return activation