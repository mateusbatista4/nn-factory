######## checando a Softmax ########

from loss import CrossEntropy, Softmax
import numpy as np


s = Softmax()
x = np.array([[0,0],
              [1,2],
              [-3,2]])

expected_softmax = np.array([[0.5       , 0.5       ],
                             [0.26894142, 0.73105858],
                             [0.00669285, 0.99330715]])
result_softmax = s(x)
assert (abs(result_softmax - expected_softmax) < 1e-8).all(), f"O resultado esperado da softmax é {expected_softmax}, mas retorna {result_softmax}"

######## checando a CrossEntropy ########

Y = np.array([[0, 1, 1],
              [1, 0, 0]])

Y_pred = np.array([[0, 1, 1],
                   [0.7, 0, 0.3],])


expected_ce = 0.1783374548265092
cross_entropy = CrossEntropy()
ce_result = cross_entropy(Y, Y_pred)
assert abs(ce_result - expected_ce) < 1e-8, f"O resultado esperado pela entropia cruzada é {expected_ce}, mas retorna {ce_result}"

expected_grad = np.array([[ 0. ,  0. ,  0. ],
                          [-0.3,  0. ,  0.3]])
grad = cross_entropy.grad(Y, Y_pred)
assert (abs(grad - expected_grad) < 1e-8).all(), f"O resultado esperado pelo gradiente da entropia cruzaada é {expected_grad}, mas retorna {grad}"