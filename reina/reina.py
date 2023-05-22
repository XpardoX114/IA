import random

# INCIALIZAMOS: Genehijo1, hijo2ramos la poblacion inicial

def initialization(num_reinas, num_agentes):
    poblacion = []
    lista = range (1, num_reinas + 1)
    for x in range (num_agentes):
        for y in range (num_reinas):
            agente = random.sample(lista, num_reinas)
        poblacion.append(agente)
    return poblacion

# MUTACION DE LOS HIJOS
def mutacion(agente):
    rand = random.randint(1, 100)
    if rand <= 45:
        num = len(agente) -1
        pos1 = random.randint(1, num)
        pos2 = random.randint(1, num)
        while (pos1 == pos2):
            pos2 = random.randint(1, num)
        agente[pos1], agente[pos2] = agente[pos2], agente[pos1]
    return agente


# CROSSOVER: Creamos a los hijos y los mutamos (50% de probabilidad)
def crossover(padre1, padre2):

    # SIGUIENTE GENERACION
    hijo1 = padre1[0:int(len(padre1)/2)]
    hijo2 = padre2[0:int(len(padre2)/2)]
    
    # GENERAMOS A LOS HIJOS
    for x in (padre1):
        if x not in (hijo2):
            hijo2.append(x)
    hijo2 = mutacion(hijo2)
    for x in (padre2):
        if x not in (hijo1):
            hijo1.append(x)
    hijo1 = mutacion(hijo2)

    return hijo1, hijo2

# REPRODUCIR: Reproducimos a toda la poblacion
def orgia(poblacion):
    new = []
    new[:] = poblacion[:]
    random.shuffle(new)
    for x in range (0, len(poblacion) -2, 2):
        new1, new2 = crossover(poblacion[x], poblacion[x+1])
        new.append(new1)
        new.append(new2)
    return new

# EVALUAMOS: Comprobamos como de buena es solución de cada cromosotra de la población
def fitness(agente):
    colq1 = 0
    colq2 = 0
    contador = 0
    for q1 in (agente):
        colq1 = 0
        for q2 in (agente):
            if abs(q1 - q2) == abs(colq2 - colq1):
                contador += 1
            colq1 += 1  
        colq2 += 1 
    return contador - len(agente)

# 
def fitpoblacion(agente):
    fit = []
    for x in agente:
        fit.append(fitness(x))
    return fit

# ORDENAR: 
def ordenar(fit, poblacion, num):
    fit, sorted_pop = zip(*sorted(zip(fit,poblacion)))
    return sorted_pop[0:num]

random.seed(20)

# MAIN MENU
num_agentes = 100
num_reinas = 12
contador = 0
poblacion = initialization(num_reinas, num_agentes)
while not fitness(poblacion[0]) == 0:
    nueva_poblacion = orgia(poblacion)
    evaluar = fitpoblacion(nueva_poblacion)
    poblacion_ordenada = ordenar(evaluar, nueva_poblacion, num_agentes)
    poblacion[:] = poblacion_ordenada[:]
    print("Fitness: ", fitness(poblacion[0]))
    contador += 1
    print("Generaciones = ",contador)
print(poblacion[0])