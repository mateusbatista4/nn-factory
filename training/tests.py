######## Checando Backward pass ########

# Arquitetura: 2 x 1 x 2
import numpy as np
from loss import ReLU, Softmax
from loss.loss import CrossEntropy
from model.model import Model
from optimizers import SGDOptimizer
from training.trainer import Trainer


m = Model([2, 1, 2], [ReLU(), Softmax()])

X = np.array([[0 ,1],
              [-1,0]])

W0 = np.array([[2],
               [1]])
b0 = np.array([[1]])
W1 = np.array([[2, 3]])
b1 = np.array([[1, -1]])

m.weights = [W0, W1]
m.bias = [b0, b1]

t = Trainer(m, None, CrossEntropy())
t.batch_size = X.shape[0]

y = np.array([[0,1],
              [1,0]])
prediction = m.forward(X)
grads = t.backward(y)

# Deixamos esse valor caso você precise verificar seus resultados
#
# expected_dZ1 = np.array([[ 0.5       , -0.5       ],
#                         [-0.11920292,  0.11920292]])
#
# expected_dZ0 = np.array([[-0.5],
#                          [ 0. ]])
#
# y_pred = np.array([[0.5       , 0.5       ],
#                    [0.88079708, 0.11920292]])


expected_dW1 = np.array([[ 0.5, -0.5]])

expected_db1 = np.array([[ 0.19039854, -0.19039854]])

expected_dW0 = np.array([[ 0.  ],
                         [-0.25]])

expected_db0 = np.array([[-0.25]])

dW1, db1 = grads[1]
assert (abs(expected_dW1 - dW1) < 1e-8).all(), f"O resultado esperado para dW1 é {expected_dW1}, mas retorna {dW1}"
assert (abs(expected_db1 - db1) < 1e-8).all(), f"O resultado esperado para  db1 é {expected_db1}, mas retorna {db1}"

dW0, db0 = grads[0]
assert (abs(expected_dW0 - dW0) < 1e-8).all(), f"O resultado esperado para  dW0 é {expected_dW0}, mas retorna {dW0}"
assert (abs(expected_db0 - db0) < 1e-8).all(), f"O resultado esperado para  db0 é {expected_db0}, mas retorna {db0}"


######## Checando o otimizador SGD ########

# Arquitetura: 2 x 1 x 2
m = Model([2, 1, 2], [ReLU(), Softmax()])

X = np.array([[0 ,1],
              [-1,0]])

W0 = np.array([[2],
               [1]])
b0 = np.array([[1]])
W1 = np.array([[2, 3]])
b1 = np.array([[1, -1]])

m.weights = [W0, W1]
m.bias = [b0, b1]

t = Trainer(m, None, CrossEntropy())
t.batch_size = X.shape[0]

y = np.array([[0,1],
              [1,0]])
prediction = m.forward(X)
grads = t.backward(y)
opt = SGDOptimizer(m, lr=1)
opt.step(grads)

expected_W0 = np.array([[2.  ],
                        [1.25]])
expected_b0 = np.array([[1.25]])

expected_W1 = np.array([[1.5, 3.5]])
expected_b1 = np.array([[ 0.80960146, -0.80960146]])

W0, b0 = m.weights[0], m.bias[0]
assert (abs(expected_W0 - W0) < 1e-8).all(), f"O resultado esperado para W0 depois do SGD atualizar o step é {expected_W0}, mas retorna {W0}"
assert (abs(expected_b0 - b0) < 1e-8).all(), f"O resultado esperado para b0 depois do SGD atualizar o step é  {expected_b0}, mas retorna {b0}"

W1, b1 = m.weights[1], m.bias[1]
assert (abs(expected_W1 - W1) < 1e-8).all(), f"O resultado esperado para W1 depois do SGD atualizar o step é  {expected_W1}, mas retorna {W1}"
assert (abs(expected_b1 - b1) < 1e-8).all(), f"O resultado esperado para b1 depois do SGD atualizar o step é  {expected_b1}, mas retorna {b1}"