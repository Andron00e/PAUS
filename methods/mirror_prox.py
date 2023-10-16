import sys
import numpy as np
from tqdm import trange

def grad_x(x, y, matrix):
    return matrix.T @ y

def grad_y(x, y, matrix):
    return matrix @ x

def prox_simplex(z, xi):
    prox = np.zeros(len(z))
    denom = 0

    for z_i, xi_i in zip(z, xi):
        denom += z_i * np.exp(-xi_i)

    for j, (z_j, xi_j) in enumerate(zip(z, xi)):
        prox[j] = (1 / denom) * z_j * np.exp(-xi_j)

    return prox

def get_gap(x, y, matrix):
    return max(matrix @ x) - min(matrix.T @ y)

def gamma_MP(c, Lipschitz):
    return c / Lipschitz

def MirrorProx(grad_x, 
               grad_y, 
               prox, 
               gamma, 
               c,
               n,
               Lipschitz,
               mean_matrix: np.ndarray, 
               max_iter = 4 * 10**3, 
               eps = 10**(-8)):
    """
    Implemetation of Mirror Prox algorithm.
    Args:
        grad_x: returns the gradient by x vector in <y,A@x>
        grad_y: returns the gradient by y
        prox_simplex: creates the probability simplex
        gamma: returns the stepsize for Mirror Prox
        c: additional constant which increases gamma if necessary
        n: number of houses
        Lipschitz: Lipschitz constant
        mean_matrix: mean matrix for the whole dataset
        max_iter: number of iterations
        eps: additional constant that allows to avoid division by zero
    """
    poli_cur = np.random.random_sample(n)
    poli_cur = poli_cur / np.linalg.norm(poli_cur, ord = 1)
    burg_cur = np.random.random_sample(n)
    burg_cur = burg_cur / np.linalg.norm(burg_cur, ord = 1)
    r_poli_cur = poli_cur
    r_burg_cur = burg_cur
    w_poli_history = []
    w_burg_history = []
    error = []
    
    for i in trange(max_iter, file = sys.stdout, ncols = 80, colour = 'cyan'):

        w_poli_cur = prox(r_poli_cur, gamma(c, Lipschitz) * grad_x(r_poli_cur, r_burg_cur, mean_matrix))
        w_burg_cur = prox(r_burg_cur, -gamma(c, Lipschitz) * grad_y(r_poli_cur, r_burg_cur, mean_matrix))
        r_poli_new = prox(r_poli_cur, gamma(c, Lipschitz) * grad_x(w_poli_cur, w_burg_cur, mean_matrix))
        r_burg_new = prox(r_burg_cur, -gamma(c, Lipschitz) * grad_y(w_poli_cur, w_burg_cur, mean_matrix))
        
        w_poli_history.append(w_poli_cur)
        w_burg_history.append(w_burg_cur)
        poli_new = sum(w_poli_history) / len(w_poli_history)
        burg_new = sum(w_burg_history) / len(w_burg_history)
        
        error.append(get_gap(poli_new, burg_new, mean_matrix))
        if error[-1] < eps:
            break
            
        r_poli_cur = r_poli_new
        r_burg_cur = r_burg_new
        poli_cur = poli_new
        burg_cur = burg_new
        
    return np.concatenate((poli_new, burg_new)), error