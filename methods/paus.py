import sys
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
from scipy.special import kl_div
from scipy.optimize import minimize
from mirror_prox import grad_x, grad_y, get_gap, gamma_MP

def get_hess(matrix):
    """
    Outpust the hessian for our special problem
    """
    hes = np.zeros((2*len(matrix), 2*len(matrix)))
    hes[:len(matrix), len(matrix):] = matrix.T
    hes[len(matrix):, :len(matrix)] = matrix
    return hes

def grads(z, matrix):
    grad_x = matrix.T @ z[1]
    grad_y = matrix @ z[0]
    return [grad_x, -1 * grad_y]

def composite_mp(z, matrix, gamma_sim, mean_matrix):
    """
    Implemetation of Composite MP algorithm from paper.
    Args:
        z: vector of two vectors [x, y]
        matrix: local mean matrix
        gamma_sim: stepsize for Composite MP for the paper calculated using similarity
        mean_matrix: mean matrix for the whole dataset
    """
    n = len(matrix)
    
    def objective_ux(ux):
        local_ux = z[1].T @ matrix @ ux
        return gamma_sim * local_ux + gamma_sim * ((grads(z, mean_matrix)[0] - grads(z, matrix)[0]).T @ ux) + np.sum(kl_div(ux, z[0]))

    def objective_uy(uy):
        local_uy = uy.T @ matrix @ z[0]
        return gamma_sim * local_uy + gamma_sim * ((grads(z, mean_matrix)[1] - grads(z, matrix)[1]).T @ uy) + np.sum(kl_div(uy, z[1]))

    constraints_ux = [{'type': 'eq', 'fun': lambda ux: np.sum(ux) - 1}, {'type': 'ineq', 'fun': lambda ux: ux}]
    constraints_uy = [{'type': 'eq', 'fun': lambda uy: np.sum(uy) - 1}, {'type': 'ineq', 'fun': lambda uy: uy}]

    result_ux = minimize(objective_ux, np.ones(n) / n, constraints=constraints_ux)
    result_uy = minimize(objective_uy, np.ones(n) / n, constraints=constraints_uy)

    return [result_ux.x, result_uy.x]

def PAUS_method(grad_x, 
                grad_y, 
                prox, 
                gamma, 
                c, 
                n,
                Lipschitz,
                gamma_sim,
                matrix, 
                mean_matrix, 
                max_iter = 4 * 10**3, 
                eps = 10**(-8)):
    """
    Implementation of Proximal Algorithm under Similarity.
    Args:
        grad_x: returns the gradient by x vector in <y,A@x>
        grad_y: returns the gradient by y
        prox: creates the probability simplex
        gamma: returns the stepsize for Mirror Prox
        c: additional constant which increases gamma if necessary
        n: number of houses
        Lipschitz: Lipschitz constant
        gamma_sim: steptize for Composite MP
        matrix: local mean matrix
        mean_matrix: mean matrix for the whole dataset
        max_iter: number of iterations
        eps: additional constant that allows to avoid division by zero
    """
    poli_cur = np.ones(n) / n
    burg_cur = np.ones(n) / n

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

        poli_tmp = sum(w_poli_history) / len(w_poli_history)
        burg_tmp = sum(w_burg_history) / len(w_burg_history)
        
        poli_new = composite_mp([poli_tmp, burg_tmp], matrix, gamma_sim, mean_matrix)[0]
        burg_new = composite_mp([poli_tmp, burg_tmp], matrix, gamma_sim, mean_matrix)[1]

        error.append(get_gap(poli_new, burg_new, mean_matrix))
        if error[-1] < eps:
            break
            
        r_poli_cur = r_poli_new
        r_burg_cur = r_burg_new
        
    return np.concatenate((poli_new, burg_new)), error