import numpy as np
import tqdm
from torch.utils.data import Dataset, DataLoader
import torch

class Trainer:
    def __init__(self, model, optimizer, loss_func):
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.batch_size = 0

    def backward(self, Y):
        """
        Argumentos:
        Y: (np.array) vetor de resultados esperados/label.

        Retorna:
        Lista de tuplas de matrizes (gradiente de pesos, gradiente de bias) ambas no formato np.array.
        A ordem desta lista deve ser igual aos pesos do modelo.
        Por exemplo: [(dW0, db0), (dW1, db1), ... ].
        """
        grads = []

        dA = self.loss_func.grad(Y, self.model.activations[-1])

        for layer_idx in range(len(self.model.layers_dims) - 1, 0, -1):
            W = self.model.weights[layer_idx - 1]
            Z = self.model.Z_list[layer_idx - 1]
            A_prev = self.model.activations[layer_idx - 1]
            dZ = dA * self.model.activation_funcs[layer_idx - 1].grad(Z)
            dW = np.dot(A_prev.T, dZ) / self.batch_size
            db = np.sum(dZ, axis=0, keepdims=True) / self.batch_size
            dA = np.dot(dZ, W.T)
            grads.append((dW, db))
        grads.reverse()
        return grads

    def train(self, n_epochs: int, train_loader: DataLoader, val_loader: DataLoader):
        """
        Argumentos:
        n_epochs: (int) número de épocas
        train_loader: (DataLoader) DataLoader de treino
        val_loader: (DataLoader) Dataloader de validação

        Retorna:
        Um dicionário com o log da função de perda do treino e da validação ao longo das épocas
        """
        log_dict = {'epoch': [],
                   'train_loss': [],
                   'val_loss': []}

        self.batch_size = train_loader.batch_size
        for epoch in tqdm.tqdm_notebook(range(n_epochs)):
            train_loss_history = []

            for i, batch in enumerate(train_loader):
                X, Y = batch
                X = X.numpy()
                Y = Y.numpy()
                Y_pred = self.model.forward(X)
                train_loss = self.loss_func(Y, Y_pred)
                train_loss_history.append(train_loss)

                grads = self.backward(Y)
                self.optimizer.step(grads)

            val_loss_history = []
            for i, batch in enumerate(val_loader):
                X, Y = batch
                X = X.numpy()
                Y = Y.numpy()
                Y_pred = self.model.forward(X)
                val_loss = self.loss_func(Y, Y_pred)
                val_loss_history.append(val_loss)

            # adicionando as losses para a história
            train_loss = np.array(train_loss_history).mean()
            val_loss = np.array(val_loss_history).mean()

            log_dict['epoch'].append(epoch)
            log_dict['train_loss'].append(train_loss)
            log_dict['val_loss'].append(val_loss)

        return log_dict