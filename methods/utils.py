import torch
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class Rademacher(sps.rv_discrete):
    def _pmf(self, x, *args):
        if x == 1:
            return 0.5
        elif x == -1:
            return 0.5
        else:
            return 0.0

    def rvs(self, size=None, random_state=None):
        if size is None:
            return super().rvs()
        elif isinstance(size, int):
            return np.random.choice([-1, 1], size=size, p=[0.5, 0.5])
        else:
            return [np.random.choice([-1, 1], size=s, p=[0.5, 0.5], random_state=random_state) for s in size]

def euclidean_distance(house1, house2):
    return np.sqrt((house1[0] - house2[0]) ** 2 + (house1[1] - house2[1]) ** 2)

def manhattan_distance(house1, house2):
  return np.abs(house1[0] - house2[0]) + np.abs(house1[1] - house2[1])

def make_house_grid(k):
    houses = [(x, y) for x in range(k) for y in range(k)]
    return houses

def get_distance_matrix(houses: list, euclidean=True):
    num_houses = len(houses)
    distance_matrix = np.zeros((num_houses, num_houses))

    for i in range(num_houses):
        for j in range(num_houses):
            distance_matrix[i][j] = euclidean_distance(houses[i], houses[j])
            if euclidean == False:
                distance_matrix[i][j] = manhattan_distance(houses[i], houses[j])

    return distance_matrix

def get_utility_matrix(distance_matrix):
    theta = 2 / distance_matrix.max()
    rad = Rademacher()
    n = len(distance_matrix)
    utility_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d = distance_matrix[i][j]
            utility_matrix[i][j] = (1 + rad.rvs(size=1)[0]) * (1 - np.exp(-theta * d))
    return utility_matrix
