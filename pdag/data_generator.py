import numpy as np
import networkx as nx
import scipy
from sys import exit

def generate_random_L(p = 10,
                      a = 0.3,
                      b = 0.7,
                      diag_a = 2,
                      diag_b = 5,
                      plot = False,
                      G = None):
    """
    randomly generates a lower triangular matrix based on a growing network graph
    Input:
        p: number of nodes
        a: lower bound for off-diagonal
        b: upper bound for off_diagonal
        diag_a: lower bound for diagonal
        diag_b: upper bound for diagonal
        G: Directed graph
    Output:
        L: Lower triangular matrix
        A: Adjacency matrix
        G: Directed graph
    """
    if(G is None):
        G = nx.gn_graph(p)
    if(nx.is_directed(G) is False):
        print('G is not directed')
        exit(1)
    ### need to relabel vertices to agree with CSCS
    mapping=dict(zip(G.nodes(),list(range(p-1,-1,-1))))
    G=nx.relabel_nodes(G,mapping)
    if(plot):
        import matplotlib.pyplot as plt
        plt.show()
        nx.draw_shell(G, with_labels=True, font_weight='bold')    
    A = nx.adjacency_matrix(G).todense()
    L = np.multiply(A,((b - a) * np.random.random_sample(size = p*p) + a).reshape(p,p))
    np.fill_diagonal(L,np.random.uniform(diag_a,diag_b,p))
    omega = np.matmul(L.T,L)
    return(omega, L, A, G)

def generate_random_B(p = 10,
                      a = 0.3,
                      b = 0.7,
                      diag_a = 2,
                      diag_b = 5,
                      plot = False,
                      G = None):
    """
    randomly generates a lower triangular matrix upto permutation based on a growing network graph
    Input:
        p: number of nodes
        a: lower bound for off-diagonal
        b: upper bound for off_diagonal
        diag_a: lower bound for diagonal
        diag_b: upper bound for diagonal
        G: Directed graph
    Output:
        L: Lower triangular matrix
        A: Adjacency matrix
        G: Directed graph
    """
    (_, L,A,G) = generate_random_L(p, a, b, diag_a, diag_b, plot, G)
    permutation = np.random.permutation(p)
    perMatrix = np.eye(p)[:,permutation]
    B = np.matmul(np.matmul(perMatrix,L),perMatrix.T)
    A = np.matmul(np.matmul(perMatrix,A),perMatrix.T)
    omega = np.matmul(B.T,B)
    return(omega, B,A,G)

def generate_random_partialB(m1,
                             m2,
                             m3,
                             p = 10,
                             a = 0.3,
                             b = 0.7,
                             diag_a = 2,
                             diag_b = 5,
                             plot = False,
                             G = None):
    """
    randomly generates a lower triangular matrix upto permutation based on a growing network graph
    Input:
        p: number of nodes
        a: lower bound for off-diagonal
        b: upper bound for off_diagonal
        diag_a: lower bound for diagonal
        diag_b: upper bound for diagonal
        G: Directed graph
    Output:
        L: Lower triangular matrix
        A: Adjacency matrix
        G: Directed graph
    """
    (_, L,A,G) = generate_random_L(p, a, b, diag_a, diag_b, plot, G)
    p1 = np.random.permutation(range(m1))
    p2 = np.random.permutation(range(m1,m2))
    p3 = np.random.permutation(range(m2,m3))
    p4 = np.random.permutation(range(m3,p))
    permutation = np.random.permutation(np.hstack([p1,p2,p3,p4]))
    perMatrix = np.eye(p)[:,permutation]
    B = np.matmul(np.matmul(perMatrix,L),perMatrix.T)
    A = np.matmul(np.matmul(perMatrix,A),perMatrix.T)
    omega = np.matmul(B.T,B)
    return(omega, B, A, G)

def generate_random_MVN_data(n, omega, mu=None):
    """generates random multivariate normal data corresponding to growing network graph
    Input:
        n: number of samples
        p: number of nodes
        a: lower bound for off-diagonal
        b: upper bound for off_diagonal
        diag_a: lower bound for diagonal
        diag_b: upper bound for diagonal
    Output:
        multivariate normal data
    """
    if mu is None:
        p = omega.shape[0]
        mu = np.zeros(p)
    cov = np.linalg.inv(omega)
    return(np.random.multivariate_normal(mu,cov,n))