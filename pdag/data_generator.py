import numpy as np
import networkx as nx
import scipy
from sys import exit


def generate_random_L(p=10,
                      a=0.3,
                      b=0.7,
                      diag_a=2,
                      diag_b=5,
                      z=0.05,
                      plot=False,
                      G=None):
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
    if (G is None):
        L = np.random.uniform(low=a, high=b, size=(p, p))
        L = L*np.tri(*L.shape)
        L = L*np.random.binomial(1, z, size=(p, p))
        if diag_a == diag_b:
            np.fill_diagonal(L, diag_a)
        else:
            np.fill_diagonal(L, np.random.uniform(diag_a, diag_b, p))
        G = nx.from_numpy_array(L.T, create_using=nx.DiGraph)
        if (plot):
            import matplotlib.pyplot as plt
            plt.show()
            nx.draw_shell(G, with_labels=True, font_weight='bold')
        A = 1.0*(L != 0)
        np.fill_diagonal(A, 0)
    else:
        if (nx.is_directed(G) is False):
            print('G is not directed')
            exit(1)
        # need to relabel vertices to agree with CSCS
        mapping = dict(zip(G.nodes(), list(range(p-1, -1, -1))))
        G = nx.relabel_nodes(G, mapping)
        if (plot):
            import matplotlib.pyplot as plt
            plt.show()
            nx.draw_shell(G, with_labels=True, font_weight='bold')
        A = nx.adjacency_matrix(G).todense()
        L = np.multiply(A, ((b - a) * np.random.random_sample(size=p*p) + a).reshape(p, p))
        if diag_a == diag_b:
            np.fill_diagonal(L, diag_a)
        else:
            np.fill_diagonal(L, np.random.uniform(diag_a, diag_b, p))
    omega = np.matmul(L.T, L)
    return (omega, L, A, G)


def generate_random_B(p=10,
                      a=0.3,
                      b=0.7,
                      diag_a=2,
                      diag_b=5,
                      z=0.05,
                      plot=False,
                      G=None):
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
    (_, L, A, G) = generate_random_L(p, a, b, diag_a, diag_b, z, plot, G)
    permutation = np.random.permutation(p)
    perMatrix = np.eye(p)[:, permutation]
    B = np.matmul(np.matmul(perMatrix, L), perMatrix.T)
    A = np.matmul(np.matmul(perMatrix, A), perMatrix.T)
    omega = np.matmul(B.T, B)
    return (omega, B, A, G)

def generate_random_partialB(p, partitions, z=0.3, lower_bound=0.1, upper_bound=0.9, seed=None, debug=False):
    """
    Generates data from a sparse multivariate normal distribution with mean 0 and variance Sigma.
    Computes sparse L, Omega, a reordered L matrix (B), adjacency matrix (A), and graph (G).

    Args:
    - p (int): Number of rows (dimension of L and Omega).
    - partitions (list): Partitioning scheme (e.g., [0, 3, 7, 15]).
    - z (float): Probability of an off-diagonal element being nonzero (sparsity control).
    - lower_bound (float): Minimum absolute value for nonzero off-diagonal elements.
    - upper_bound (float): Maximum absolute value for nonzero off-diagonal elements.
    - seed (int, optional): Random seed for reproducibility.

    Returns:
    - omega (numpy.ndarray): Inverse covariance matrix (Ω = B^T B).
    - B (numpy.ndarray): Lower triangular matrix L with reordered rows and columns.
    - A (numpy.ndarray): Adjacency matrix of B (where B[i, j] ≠ 0).
    - G (networkx.DiGraph): Directed graph representation of adjacency A.
    """
    if seed is not None:
        np.random.seed(seed)

    # Step 1: Generate a sparse lower triangular matrix L
    L = np.zeros((p, p))  # Initialize L with zeros
    mask = np.tril(np.random.binomial(1, z, size=(p, p)), k=-1)  # Lower triangle sparsity mask (excluding diagonal)
    values = np.random.uniform(lower_bound, upper_bound, size=(p, p)) * mask  # Generate random values within bounds

    np.fill_diagonal(L, 1)  # Set diagonal elements to 1
    L += values  # Apply generated values to L

    # Step 2: Generate a partition-wise row & column permutation
    permuted_indices = np.zeros(p, dtype=int)
    start_idx = 0
    for i in range(len(partitions) + 1):
        if i == 0:
            block_start = 0 
        else:
            block_start = partitions[i-1]
        if i == len(partitions):
            block_end = p
        else:
            block_end = partitions[i]
        block_permutation = np.random.permutation(np.arange(block_start, block_end))  # Independent permutation in block
        permuted_indices[start_idx:(start_idx + len(block_permutation))] = block_permutation
        start_idx += len(block_permutation)

    if debug:
        print('permuted_indices: ', permuted_indices)

    # Step 3: Apply the permutation to BOTH rows and columns to get B
    B = L[permuted_indices, :]  # Reorder rows
    B = B[:, permuted_indices]  # Reorder columns

    # Step 4: Compute Omega (Inverse covariance matrix)
    omega = B.T @ B  # Omega = B^T B

    # Step 5: Compute adjacency matrix A from B (nonzero entries)
    A = (np.abs(B) > 1e-5).astype(int)  # Adjacency based on nonzero entries in B
    A_copy = A.copy()
    np.fill_diagonal(A_copy, 0)  # Set diagonal to 0
    A_copy = (np.abs(A_copy) > 1e-5)  # Compare absolute values with 0.00001

    # Step 6: Create directed graph G using adjacency matrix A
    G = nx.from_numpy_array(A_copy.T, create_using=nx.DiGraph)

    return (omega, B, A, G)




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
    return (np.random.multivariate_normal(mu, cov, n))
