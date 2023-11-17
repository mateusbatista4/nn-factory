import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch

import matplotlib.pyplot as plt
from loss import ReLU, Softmax
from loss.loss import CrossEntropy

from model import Model
from optimizers import SGDOptimizer
from training.trainer import Trainer


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import seaborn as sn

# Módulos de preparação de dados para serem usados pelo Pytorch Dataloader

class CIFAR10(Dataset):
    def __init__(self, x, y=None, transform=None):
        self._x = x
        self._y = y#.squeeze() if y is not None else None
        self._transform = transform

    def __len__(self):
        return len(self._x)

    def __getitem__(self, idx):
        image = self._x[idx]
        if self._transform is not None:
            image = self._transform(image)

        image = image.flatten()
        if self._y is None:
            return image

        # one hot encoding
        label = [0] * 10
        label[self._y[idx]] = 1
        return image, torch.Tensor(label)
    
# um exemplo de uma função de normalização, mas você poderá implementar outras
def normalize(X):
    return (X - X.mean())/(X.std() + 1e-8)

# Função auxiliar para traçar perdas ao longo das épocas
def plot_history(history, method= ""):
    """
    Plote do histórico de loss (função de perda)
    """
    fig, ax = plt.subplots()
    ax.plot(history['train_loss'], color='#407cdb', label='Train')
    ax.plot(history['val_loss'],color='#db5740', label='Validation')

    ax.legend(loc='upper left')
    handles, labels = ax.get_legend_handles_labels()
    lgd = dict(zip(labels, handles))
    ax.legend(lgd.values(), lgd.keys())

    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.title('Loss ao longo das épocas ' + method)
    plt.show()
    
def evaluate(model, dataloader : DataLoader ):

    predictions = []
    expected = []

    for batch in dataloader:
      X, Y = batch
      X = X.numpy()
      Y = Y.numpy()

      predictions.append(model.forward(X))
      expected.append(Y)

    predictions = np.vstack(predictions)
    expected = np.vstack(expected)
    return predictions, expected

def conf_matrix(predicted, expected):


    cm = confusion_matrix(expected.argmax(axis=1), predicted.argmax(axis=1))

    #cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    df_cm = pd.DataFrame(cm, index = [i for i in "ABCDEFGHIJ"],
                  columns = [i for i in "ABCDEFGHIJ"])

    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True, cmap="Greens")
     # Calcular métricas
    print("Classification Report:")
    target_names = [f"Class {i}" for i in range(10)]
    print(classification_report(expected.argmax(axis=1), predicted.argmax(axis=1), target_names=target_names))
    
## Run the following script to get dataset:
## !wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1fdyH-HvuqypF5yMdOz4_EUWGxgGL9Rwq' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1fdyH-HvuqypF5yMdOz4_EUWGxgGL9Rwq" -O 'dataset.npy' && rm -rf /tmp/cookies.txt

# defina o caminho do conjunto de dados corretamente
dataset_path = 'dataset.npy'
dataset = np.load(dataset_path, allow_pickle=True).item()

x_train, y_train = dataset['train_images'], dataset['train_labels']
x_val, y_val = dataset['val_images'], dataset['val_labels']
x_test, y_test = dataset['test_images'], dataset['test_labels']

train_set = CIFAR10(x_train, y_train, transform=normalize)
val_set = CIFAR10(x_val, y_val, transform=normalize)
test_set = CIFAR10(x_test, y_test, transform=normalize)

train_loader = DataLoader(train_set, batch_size=100, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32)
test_loader = DataLoader(test_set, batch_size=32)


## Creating a model and training it:
model = Model([3072, 500, 100, 10], [ReLU(), ReLU(), Softmax()], initialization_method="random")
opt = SGDOptimizer(model, lr=1e-5)
trainer = Trainer(model, opt, CrossEntropy())
history = trainer.train(15, train_loader, val_loader)
plot_history(history)


