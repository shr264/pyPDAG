from numba import jit, njit, prange
import numpy as np
import networkx as nx

np.random.seed(seed=3689)

def show_graph_with_labels(adjacency_matrix):
    import matplotlib.pyplot as plt
    p = len(adjacency_matrix)
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.DiGraph()
    gr.add_edges_from(edges)
    nx.draw(gr, node_size=150)
    plt.show()

@njit(fastmath=True)
def _softhresh(x,l):
    return np.sign(x)*max(abs(x)-l,0)

@njit(fastmath=True)
def _Bii(S,B,i):
    p = len(B)
    sumterm = 0
    for j in range(p):
        if j != i:
            sumterm += S[i][j]*B[i][j]
    out = (-1*sumterm + np.sqrt(1*sumterm**2+4*S[i][i]))/(2*S[i][i])
    return out

@njit(fastmath=True)
def _Bik(S,B,i,k,l):
    out = 0
    for j in range(B.shape[0]):
        if j != k:
            out += -2*(S[k][j]*B[i][j])/(4*S[k][k])
    out = _softhresh(out, l/(4*S[k][k]) )
    return out

@njit(fastmath=True)
def _offDiagBlock(B,S,l,m1,m2,p):

    for i in range((m1+1),(m2),1):
        for k in range(m1):
            B[k][i] = 0
            B[i][k] = _Bik(S,B,i,k,l)
    return(B)

@njit(fastmath=True)
def _isCyclicUtil(v, visited, recStack, graph, p):
    visited[v] = True
    recStack[v] = True

    for neighbour in range(p):
        if neighbour != v and graph[v][neighbour] == 1:
            if visited[neighbour] == False:
                if _isCyclicUtil(neighbour, visited, recStack, graph, p) == True:
                    return True
            elif recStack[neighbour] == True:
                return True

    recStack[v] = False
    return False

@njit(fastmath=True)
def _isCyclic(B, i, j):
    graph = 1*(B > 0)
    graph[i][j] = 1
    np.fill_diagonal(graph, 0)
    p = len(B)
    visited = [False]*p
    recStack = [False]*p
    for ii in range(p):
        if visited[ii] == False:
            if _isCyclicUtil(ii, visited, recStack, graph, p):
                return True
    return False

@njit(fastmath=True)
def _qBki(S,Bki,B,k,i,l):
    p = len(S)
    sumterm = 0
    for j in range(p):
        if j != i:
            sumterm += S[i][j]*B[k][j]
    out1 = S[i][i]*Bki**2 + 2*Bki*sumterm + l*abs(Bki)
    out2 = 0
    if(out1<out2):
        return(out1,Bki)
    else:
        return(0, 0)

@njit(fastmath=True)
def _diagonalBlock(B, S, l, m1, m2, p):
    for i in range(m1,m2):
        B[i][i] = _Bii(S,B,i)
    for i in range(m1+1,m2):
        for k in range(m1,i):
            #print(i,k)
            if _isCyclic(B, i, k): # adding edge from k -> i cases cycle
                B[k][i] = 0 # set B_ki = 0
                B[i][k] = _Bik(S,B,i,k,l) # update B_ik
            elif _isCyclic(B, k, i): # adding edge from k -> i cases cycle
                B[i][k] = 0 # set B_ki = 0
                B[k][i] = _Bik(S,B,k,i,l) # update B_ik   
            else:
                Bik = _Bik(S,B,i,k,l)
                Bki = _Bik(S,B,k,i,l)
                qbki = _qBki(S,Bki,B,k,i,l)
                qbik = _qBki(S,Bik,B,i,k,l)
                if qbki[0] < qbik[0]:
                    B[k][i] = qbki[1]
                    B[i][k] = 0
                elif qbki[0] > qbik[0]:
                    B[i][k] = qbik[1]
                    B[k][i] = 0
                else:
                    u = np.random.uniform(0,1)
                    if u < 0.5:
                        B[k][i] = Bki
                        B[i][k] = 0
                    else:
                        B[i][k] = Bki
                        B[k][i] = 0
    return B

@njit(fastmath=True)
def _blockPartial2(blocknum, B, S, l, m1, p, Bnew):
    if blocknum == 0:
        temp = _diagonalBlock(B,S,l,0,m1,p)
        Bnew[0:m1][0:m1] = temp[0:m1][0:m1]
    elif blocknum == 1:
        temp = _offDiagBlock(B,S,l,m1,p,p)
        temp = _diagonalBlock(temp,S,l,m1,p,p)
        Bnew[m1:p][0:p] = temp[m1:p][0:p]

@njit(parallel=True)
def partial2(X, l, m1, eps = 10^(-4), maxitr = 10, init=None):
    n = X.shape[0]
    p = X.shape[1]
    
    S = np.dot(X.T,X)/n
    if init is None:
        B = Bold = np.eye(p)
    else:
        B = Bold = init
        
    diff = 1
    itr = 1

    while( ( diff > eps ) and ( itr < maxitr ) ):  
        Bnew = B
          
        for blocknum in prange(2):
            _blockPartial2(blocknum, B, S, l, m1, p, Bnew)

        B = Bnew
        diff = np.max(B-Bold)
        Bold = B
        itr = itr +1
    
    return B

njit(fastmath=True)
def _blockPartial3(blocknum, B, S, l, m1, m2, p, Bnew):
    if blocknum == 0:
        temp = _diagonalBlock(B,S,l,0,m1,p)
        Bnew[0:m1][0:m1] = temp[0:m1][0:m1]
    elif blocknum == 1:
        temp = _offDiagBlock(B,S,l,m1,m2,p)
        temp = _diagonalBlock(temp,S,l,m1,m2,p)
        Bnew[m1:m2][0:m2] = temp[m1:m2][0:m2]
    elif blocknum == 2:
        temp = _offDiagBlock(B,S,l,m2,p,p)
        temp = _diagonalBlock(temp,S,l,m2,p,p)
        Bnew[m2:p][0:p] = temp[m2:p][0:p]

@njit(parallel=True)
def partial3(X, l, m1, m2, eps = 10^(-4), maxitr = 10, init=None):
    n = X.shape[0]
    p = X.shape[1]
    
    S = np.dot(X.T,X)/n
    if init is None:
        B = Bold = np.eye(p)
    else:
        B = Bold = init
        
    diff = 1
    itr = 1

    while( ( diff > eps ) and ( itr < maxitr ) ):  
        Bnew = B
          
        for blocknum in prange(4):
            _blockPartial3(blocknum, B, S, l, m1, m2, p, Bnew)

        B = Bnew
        diff = np.max(B-Bold)
        Bold = B
        itr = itr +1
    
    return B

@njit(fastmath=True)
def _blockPartial4(blocknum, B, S, l, m1, m2, m3, p, Bnew):
    if blocknum == 0:
        temp = _diagonalBlock(B,S,l,0,m1,p)
        Bnew[0:m1][0:m1] = temp[0:m1][0:m1]
    elif blocknum == 1:
        temp = _offDiagBlock(B,S,l,m1,m2,p)
        temp = _diagonalBlock(temp,S,l,m1,m2,p)
        Bnew[m1:m2][0:m2] = temp[m1:m2][0:m2]
    elif blocknum == 2:
        temp = _offDiagBlock(B,S,l,m2,m3,p)
        temp = _diagonalBlock(temp,S,l,m2,m3,p)
        Bnew[m2:m3][0:m3] = temp[m2:m3][0:m3]
    elif blocknum == 3:
        temp = _offDiagBlock(B,S,l,m3,p,p)
        temp = _diagonalBlock(temp,S,l,m3,p,p)
        Bnew[m3:p][0:p] = temp[m3:p][0:p]

@njit(parallel=True)
def partial4(X, l, m1, m2, m3, eps = 10^(-4), maxitr = 10, init=None):
    n = X.shape[0]
    p = X.shape[1]
    
    S = np.dot(X.T,X)/n
    if init is None:
        B = Bold = np.eye(p)
    else:
        B = Bold = init
        
    diff = 1
    itr = 1

    while( ( diff > eps ) and ( itr < maxitr ) ):  
        Bnew = B
          
        for blocknum in prange(4):
            _blockPartial4(blocknum, B, S, l, m1, m2, m3, p, Bnew)

        B = Bnew
        diff = np.max(B-Bold)
        Bold = B
        itr = itr +1
    
    return B

class PDAG:
    def __init__(self, partitions):
        self.partitions = partitions
        self.Bhat = None
        assert self.partitions <= 4, 'Higher partitions have not been implemented yet'

    def fit(self,X, l, m1=None, m2=None, m3=None, eps = 10^(-4), maxitr = 10, init=None):
        if self.partitions == 2:
            assert isinstance(m1, int), 'm1 must be an integer'
            self.Bhat = partial2(X, 0.1, m1, eps, maxitr, init)
        elif self.partitions == 3:
            assert isinstance(m1, int) and isinstance(m2, int), 'm1 and m2 must be integers'
            self.Bhat = partial3(X, 0.1, m1, m2, eps, maxitr, init)
        elif self.partitions == 4:
            assert isinstance(m1, int) and isinstance(m2, int) and isinstance(m3, int), 'm1, m2 and m3 must be integers'
            self.Bhat = partial4(X, 0.1, m1, m2, m3, eps, maxitr, init)
        return self.Bhat

    def getAdjacency(self):
        A = (self.Bhat != 0)*1.0
        return A

    def getGraph(self):
        G = nx.from_numpy_matrix(self.Bhat, create_using=nx.DiGraph())
        return G

    def plotGraph(self):
        adj = self.getAdjacency()
        show_graph_with_labels(adj)
