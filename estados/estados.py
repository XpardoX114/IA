# input = 1 --> cambiar estado
# clock = que mire la entrada, que la cambie, que mire la salida
# 

import time
import random

class Estado():
    def __init__(self, estado, vector_prob):
        self.estado = estado
        self.vector_prob = vector_prob

    def update(self, matrix):
        dato = random.random()
        if dato <= self.vector_prob[0]:
            new_state = "sol"
            vec_prob = matrix[0]
        elif dato <= self.vector_prob[0] + self.vector_prob[1]:
            new_state = "nube"
            vec_prob = matrix[1]
        else:
            new_state = "lluvia"
            vec_prob = matrix[2]
        return Estado(new_state, vec_prob)

    def show(self):
        print(self.estado)

sol = [0.7, 0.3, 0.1]
nube = [0.4, 0.3, 0.]
lluvia = [0.5, 0.3, 0.2]
state_matrix = [sol, nube, lluvia]

estado_actual = Estado("sol", sol)

while True:
    estado_actual.show()
    estado_actual = estado_actual.update(state_matrix)
    time.sleep(1)