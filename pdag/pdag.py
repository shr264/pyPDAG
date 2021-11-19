from numba import njit, prange
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

def standardize(A):
    A = (A - np.mean(A, axis=0)) / np.std(A, axis=0)
    return A

@njit(fastmath=True)
def _softhresh(x,l):
    return np.sign(x)*max(abs(x)-l,0)

@njit(fastmath=True)
def _getActiveSet(B):
    _active_set = np.nonzero(B)
    _active_set2 = np.column_stack((_active_set[0],_active_set[1]))
    return _active_set2

@njit(fastmath=True)
def _boundBii(out):
    return min( abs(out) , 1.0 )

@njit(fastmath=True)
def _Bii(S,B,k):
    p = len(B)
    sumterm = 0
    for j in range(p):
        if j != k:
            sumterm += S[k][j]*B[k][j]
    out = (-1*sumterm + np.sqrt(1*sumterm**2+4*S[k][k]))/(2*S[k][k])
    return _boundBii(out)

@njit(fastmath=True)
def _boundBik(out, B):
    if out > 0:
        return min(out, 1.0 )
    else:
        return max(out, - 1.0 )

@njit(fastmath=True)
def _Bik(S,B,i,j,l):
    sumterm = 0
    for k in range(B.shape[0]):
        if k != j:
            #out += -2*(S[k][j]*B[i][j])/(4*S[k][k])
            #sumterm += (-1)*(S[j][k]*B[i][k])/(2*S[j][j])
            sumterm += (S[j][k]*B[i][k])
    out = -sumterm/(2*S[j][j])
    out = _softhresh(out, l/(4*S[j][j]) )
    if out != 0:
        out = _boundBik(out, B)
    return out

@njit(fastmath=True)
def _offDiagBlock(B,S,l,m1,m2,p):

    for i in range((m1+1),(m2),1):
        for k in range(m1):
            #print(i,k)
            B[k][i] = 0
            B[i][k] = _Bik(S,B,i,k,l)
    return(B)

@njit(fastmath=True)
def _offDiagBlockActive(B, S, l, m1, m2, p, active_set):
    for i, k in active_set:
        if m1+1 <= i < m2 and 0 <= k < m2:
            B[k][i] = 0
            B[i][k] = _Bik(S,B,i,k,l)
    return(B)

@njit(fastmath=True)
def _isCyclicUtil(v, visited, recStack, graph, p, m1, m2):
    visited[v] = True
    recStack[v] = True

    for neighbour in range(m1, m2):
        if neighbour != v and graph[v][neighbour] == 1:
            if visited[neighbour] == False:
                if _isCyclicUtil(neighbour, visited, recStack, graph, p, m1, m2) == True:
                    return True
            elif recStack[neighbour] == True:
                return True

    recStack[v] = False
    return False

@njit(fastmath=True)
def _isCyclic(B, i, j, m1, m2):
    graph = 1*(B != 0)
    graph[i][j] = 1
    np.fill_diagonal(graph, 0)
    p = len(B)
    visited = [False]*p
    recStack = [False]*p
    # can we just do i and j here?
    if visited[j] == False:
        if _isCyclicUtil(j, visited, recStack, graph, p, m1, m2):
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
def _qBij(S,Bij,B,i,j,l):
    p = len(S)
    sumterm = 0
    for k in range(p):
        if k != j:
            sumterm += S[j][k]*B[i][k]
    out1 = S[j][j]*Bij**2 + 2*Bij*sumterm + l*abs(Bij)
    out2 = 0
    if(out1<out2):
        return(out1,Bij)
    else:
        return(0, 0)

@njit(fastmath=True)
def _diagonalBlock(B, S, l, m1, m2, p):
    for i in range(m1,m2):
        B[i][i] = _Bii(S,B,i)
    for i in range(m1+1,m2):
        for k in range(m1,i):
            #print(i,k)
            if _isCyclic(B, i, k, m1, m2): # adding edge from k -> i cases cycle
                #print(f'cycle case: adding edge from {k} -> {i} cases cycle')
                #B[k][i] = 0 # set B_ki = 0
                #B[i][k] = _Bik(S,B,i,k,l) # update B_ik
                B[i][k] = 0 # set B_ki = 0
                B[k][i] = _Bik(S,B,k,i,l) # update B_ik   
            elif _isCyclic(B, k, i, m1, m2): # adding edge from i -> k cases cycle
                #print(f'cycle case: adding edge from {i} -> {k} cases cycle')
                #B[i][k] = 0 # set B_ki = 0
                #B[k][i] = _Bik(S,B,k,i,l) # update B_ik   
                B[k][i] = 0 # set B_ki = 0
                B[i][k] = _Bik(S,B,i,k,l) # update B_ik
            else:
                #print(f'no cycle case', end= ' \t')
                Bik = _Bik(S,B,i,k,l)
                Bki = _Bik(S,B,k,i,l)
                qbki = _qBij(S,Bki,B,k,i,l)
                qbik = _qBij(S,Bik,B,i,k,l)
                if qbki[0] > qbik[0]:
                    #print(f'adding edge from {i} -> {k} ', qbki[0] , qbik[0], qbki[1] , qbik[1], end = '\t')
                    #B[k][i] = qbki[1]
                    #B[i][k] = 0
                    B[i][k] = qbik[1]
                    B[k][i] = 0
                elif qbki[0] < qbik[0]:
                    #print(f'adding edge from {k} -> {i} ', qbki[0] , qbik[0], qbki[1] , qbik[1], end = '\t')
                    #B[i][k] = qbik[1]
                    #B[k][i] = 0
                    B[k][i] = qbki[1]
                    B[i][k] = 0
                else:
                    #print('random case', end= ' \t')
                    u = np.random.uniform(0,1)
                    if u < 0.5:
                        #B[k][i] = qbki[1]
                        #B[i][k] = 0
                        B[i][k] = qbik[1]
                        B[k][i] = 0
                    else:
                        #B[i][k] = qbik[1]
                        #B[k][i] = 0
                        B[k][i] = qbki[1]
                        B[i][k] = 0
                #print('')
    return B

@njit(parallel=True)
def _rowBlock(B, S, l, m1, m2, p):
    for i in range(m1,m2):
        B[i][i] = _Bii(S,B,i)
    for i in range((m1+1),(m2),1):
        for k in range(i):
            if k < m1:
            #print(i,k)
                B[k][i] = 0
                B[i][k] = _Bik(S,B,i,k,l)
            else:
                #print(i,k)
                if _isCyclic(B, i, k, m1, m2): # adding edge from k -> i cases cycle
                    #print(f'cycle case: adding edge from {k} -> {i} cases cycle')
                    B[i][k] = 0 # set B_ki = 0
                    B[k][i] = _Bik(S,B,k,i,l) # update B_ik   
                elif _isCyclic(B, k, i, m1, m2): # adding edge from i -> k cases cycle
                    #print(f'cycle case: adding edge from {i} -> {k} cases cycle') 
                    B[k][i] = 0 # set B_ki = 0
                    B[i][k] = _Bik(S,B,i,k,l) # update B_ik
                else:
                    #print(f'no cycle case', end= ' \t')
                    Bik = _Bik(S,B,i,k,l)
                    Bki = _Bik(S,B,k,i,l)
                    qbki = _qBij(S,Bki,B,k,i,l)
                    qbik = _qBij(S,Bik,B,i,k,l)
                    if qbki[0] > qbik[0]:
                        B[i][k] = qbik[1]
                        B[k][i] = 0
                    elif qbki[0] < qbik[0]:
                        B[k][i] = qbki[1]
                        B[i][k] = 0
                    else:
                        #print('random case', end= ' \t')
                        u = np.random.uniform(0,1)
                        if u < 0.5:
                            B[i][k] = qbik[1]
                            B[k][i] = 0
                        else:
                            B[k][i] = qbki[1]
                            B[i][k] = 0
                    #print('')
    return B


@njit(fastmath=True)
def _diagonalBlockActive(B, S, l, m1, m2, p, active_set):
    for i in range(m1,m2):
        B[i][i] = _Bii(S,B,i)
    for i, k in active_set:
        if m1+1 <= i < m2 and m1 <= k < i:
            if _isCyclic(B, i, k, m1, m2): # adding edge from k -> i cases cycle
                #print(f'cycle case: adding edge from {k} -> {i} cases cycle')
                #B[k][i] = 0 # set B_ki = 0
                #B[i][k] = _Bik(S,B,i,k,l) # update B_ik
                B[i][k] = 0 # set B_ki = 0
                B[k][i] = _Bik(S,B,k,i,l) # update B_ik   
            elif _isCyclic(B, k, i, m1, m2): # adding edge from i -> k cases cycle
                #print(f'cycle case: adding edge from {i} -> {k} cases cycle')
                #B[i][k] = 0 # set B_ki = 0
                #B[k][i] = _Bik(S,B,k,i,l) # update B_ik   
                B[k][i] = 0 # set B_ki = 0
                B[i][k] = _Bik(S,B,i,k,l) # update B_ik
            else:
                #print(f'no cycle case', end= ' \t')
                Bik = _Bik(S,B,i,k,l)
                Bki = _Bik(S,B,k,i,l)
                qbki = _qBij(S,Bki,B,k,i,l)
                qbik = _qBij(S,Bik,B,i,k,l)
                if qbki[0] > qbik[0]:
                    #print(f'adding edge from {i} -> {k} ', qbki[0] , qbik[0], qbki[1] , qbik[1], end = '\t')
                    #B[k][i] = qbki[1]
                    #B[i][k] = 0
                    B[i][k] = qbik[1]
                    B[k][i] = 0
                elif qbki[0] < qbik[0]:
                    #print(f'adding edge from {k} -> {i} ', qbki[0] , qbik[0], qbki[1] , qbik[1], end = '\t')
                    #B[i][k] = qbik[1]
                    #B[k][i] = 0
                    B[k][i] = qbki[1]
                    B[i][k] = 0
                else:
                    #print('random case', end= ' \t')
                    u = np.random.uniform(0,1)
                    if u < 0.5:
                        #B[k][i] = qbki[1]
                        #B[i][k] = 0
                        B[i][k] = qbik[1]
                        B[k][i] = 0
                    else:
                        #B[i][k] = qbik[1]
                        #B[k][i] = 0
                        B[k][i] = qbki[1]
                        B[i][k] = 0
                #print('')
    return B

@njit(parallel=True)
def _rowBlockActive(B, S, l, m1, m2, p, active_set):
    for i in range(m1,m2):
        B[i][i] = _Bii(S,B,i)
    for i, k in active_set:
        if m1+1 <= i < m2 and 0 <= k < i:
            if k < m1:
            #print(i,k)
                B[k][i] = 0
                B[i][k] = _Bik(S,B,i,k,l)
            else:
                #print(i,k)
                if _isCyclic(B, i, k, m1, m2): # adding edge from k -> i cases cycle
                    B[i][k] = 0 # set B_ki = 0
                    B[k][i] = _Bik(S,B,k,i,l) # update B_ik   
                elif _isCyclic(B, k, i, m1, m2): # adding edge from i -> k cases cycle 
                    B[k][i] = 0 # set B_ki = 0
                    B[i][k] = _Bik(S,B,i,k,l) # update B_ik
                else:
                    #print(f'no cycle case', end= ' \t')
                    Bik = _Bik(S,B,i,k,l)
                    Bki = _Bik(S,B,k,i,l)
                    qbki = _qBij(S,Bki,B,k,i,l)
                    qbik = _qBij(S,Bik,B,i,k,l)
                    if qbki[0] > qbik[0]:
                        B[i][k] = qbik[1]
                        B[k][i] = 0
                    elif qbki[0] < qbik[0]:
                        B[k][i] = qbki[1]
                        B[i][k] = 0
                    else:
                        u = np.random.uniform(0,1)
                        if u < 0.5:
                            B[i][k] = qbik[1]
                            B[k][i] = 0
                        else:
                            B[k][i] = qbki[1]
                            B[i][k] = 0
                    #print('')
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
    #print(temp, Bnew)
    return Bnew

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

    while( ( diff > eps ) and ( itr <= maxitr ) ):  
        #print(B)
        Bnew = B
          
        for blocknum in prange(2):
        #for blocknum in range(2):
            _blockPartial2(blocknum, B, S, l, m1, p, Bnew)

        B = Bnew
        diff = np.max(B-Bold)
        Bold = B
        itr = itr +1
        #print('-'*30)
    return B

@njit(fastmath=True)
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

    while( ( diff > eps ) and ( itr <= maxitr ) ):  
        Bnew = B
          
        for blocknum in prange(3):
            _blockPartial3(blocknum, B, S, l, m1, m2, p, Bnew)

        B = Bnew
        diff = np.max(B-Bold)
        Bold = B
        itr = itr +1
    
    return B

@njit(fastmath=True)
def _blockPartial4NA(blocknum, B, S, l, m1, m2, m3, p, Bnew):
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
def partial4NA(X, l, m1, m2, m3, eps = 10^(-4), maxitr = 10, init=None):
    n = X.shape[0]
    p = X.shape[1]
    
    S = np.dot(X.T,X)/n
    if init is None:
        B = Bold = np.eye(p)
    else:
        B = Bold = init
        
    diff = 1
    itr = 1

    while( ( diff > eps ) and ( itr <= maxitr ) ):  
        Bnew = B
          
        for blocknum in prange(4):
            _blockPartial4NA(blocknum, B, S, l, m1, m2, m3, p, Bnew)

        B = Bnew
        diff = np.max(B-Bold)
        Bold = B
        itr = itr +1
    
    return B

@njit(fastmath=True)
def _blockPartial4(blocknum, B, S, l, m1, m2, m3, p, Bnew, active, active_set):
    if not active:
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
    else:
        if blocknum == 0:
            temp = _diagonalBlockActive(B,S,l,0,m1,p, active_set)
            Bnew[0:m1][0:m1] = temp[0:m1][0:m1]
        elif blocknum == 1:
            temp = _offDiagBlockActive(B,S,l,m1,m2,p, active_set)
            temp = _diagonalBlockActive(temp,S,l,m1,m2,p, active_set)
            Bnew[m1:m2][0:m2] = temp[m1:m2][0:m2]
        elif blocknum == 2:
            temp = _offDiagBlockActive(B,S,l,m2,m3,p, active_set)
            temp = _diagonalBlockActive(temp,S,l,m2,m3,p, active_set)
            Bnew[m2:m3][0:m3] = temp[m2:m3][0:m3]
        elif blocknum == 3:
            temp = _offDiagBlockActive(B,S,l,m3,p,p, active_set)
            temp = _diagonalBlockActive(temp,S,l,m3,p,p, active_set)
            Bnew[m3:p][0:p] = temp[m3:p][0:p]

@njit(parallel=True)
def partial4(X, l, m1, m2, m3, eps = 10^(-4), max_itr = 10, max_active_itr = 4, init=None):
    n = X.shape[0]
    p = X.shape[1]
    
    S = np.dot(X.T,X)/n
    if init is None:
        B = Bold = np.eye(p)
    else:
        B = Bold = init
        
    diff = 1
    itr = 1
    active = False
    active_itr = 1
    active_set_condition = True

    while (active_itr==1) or (active_set_condition and active_itr <= max_active_itr):

        old_active_set = new_active_set = _getActiveSet(B)
        while( ( diff > eps ) and ( itr < max_itr ) ):  
            #print("Active itr: ", active_itr, " itr: " , itr)
            Bnew = B  
            for blocknum in prange(4):
                _blockPartial4(blocknum, B, S, l, m1, m2, m3, p, Bnew, active, new_active_set)
            B = Bnew

            diff = np.max(B-Bold)
            Bold = B
            itr = itr + 1
            if not active:
                active = True
                new_active_set = _getActiveSet(B)

        if new_active_set.shape[0] == old_active_set.shape[0] and new_active_set.shape[1] == old_active_set.shape[1]:
            if (new_active_set == old_active_set).all():
                active_set_condition = False

        active = False
        active_itr += 1
        itr = 1
    
    return B


@njit(fastmath=True)
def _blockPartial(blocknum, B, S, l, partition, p, eps, maxitr):
    #print(blocknum)
    Bold = B
    diff = 1
    itr = 1
    #partition = [m1, m2, m3] => 4 blocks
    num_blocks = len(partition) 
    while( ( diff > eps ) and ( itr <= maxitr ) ): 
        if blocknum == 0 and partition.size == 0:
            B = _diagonalBlock(B,S,l,0,p,p) 
        elif blocknum == 0 and partition.size != 0:
            B = _diagonalBlock(B,S,l,0,partition[blocknum],p) 
        elif blocknum > 0 and blocknum < num_blocks:   # blocknum = 1 or 2
            B = _offDiagBlock(B,S,l,partition[blocknum-1],partition[blocknum],p)
            B = _diagonalBlock(B,S,l,partition[blocknum-1],partition[blocknum],p)
        elif blocknum == num_blocks:    # blocknum = 3
            B = _offDiagBlock(B,S,l,partition[-1],p,p)
            B = _diagonalBlock(B,S,l,partition[-1],p,p)
        diff = np.max(B-Bold)
        Bold = B
        itr = itr +1
    #print(B)
    #print('-'*20)
    return B


@njit(fastmath=True)
def _blockPartialActive(blocknum, B, S, l, partition, p, eps, maxitr, active_set):
    #print(blocknum)
    Bold = B
    diff = 1
    itr = 1
    #partition = [m1, m2, m3] => 4 blocks
    num_blocks = len(partition) 
    while( ( diff > eps ) and ( itr <= maxitr ) ):  
        if blocknum == 0:
            B = _diagonalBlockActive(B,S,l,0,partition[blocknum],p, active_set) 
        elif blocknum > 0 and blocknum < num_blocks:   # blocknum = 1 or 2
            B = _offDiagBlockActive(B,S,l,partition[blocknum-1],partition[blocknum],p, active_set)
            B = _diagonalBlockActive(B,S,l,partition[blocknum-1],partition[blocknum],p, active_set)
        elif blocknum == num_blocks:    # blocknum = 3
            B = _offDiagBlockActive(B,S,l,partition[-1],p,p, active_set)
            B = _diagonalBlockActive(B,S,l,partition[-1],p,p, active_set)
        diff = np.max(B-Bold)
        Bold = B
        itr = itr + 1
    #print(B)
    #print('-'*20)
    return B


@njit(fastmath=True)
def partial(X, l, partition, eps = 10^(-4), maxitr = 10, init=None):
    n = X.shape[0]
    p = X.shape[1]
    S = np.dot(X.T,X)/n
    num_blocks = len(partition)
    if init is None:
        B = Bold = np.eye(p)
    else:
        B = Bold = init
        
    for blocknum in prange(num_blocks+1):
        if blocknum == 0:
            B[0:partition[blocknum]][0:partition[blocknum]] = _blockPartial(blocknum, B, S, l, partition, p, eps, maxitr)[0:partition[blocknum]][0:partition[blocknum]]
        elif blocknum > 0 and blocknum < num_blocks:
            B[partition[blocknum-1]:partition[blocknum]][partition[blocknum-1]:partition[blocknum]] = _blockPartial(blocknum, B, S, l, partition, p, eps, maxitr)[partition[blocknum-1]:partition[blocknum]][partition[blocknum-1]:partition[blocknum]]
        elif blocknum == num_blocks:
            B[partition[blocknum-1]:p][partition[blocknum-1]:p] = _blockPartial(blocknum, B, S, l, partition, p, eps, maxitr)[partition[blocknum-1]:p][partition[blocknum-1]:p]
    return B

@njit(parallel=True)
def partialActive(X, l, partition, eps = 10^(-4), maxitr = 10, max_active_itr = 10, init=None):
    n = X.shape[0]
    p = X.shape[1]
    S = np.dot(X.T,X)/n
    num_blocks = len(partition)
    if init is None:
        B = Bold = np.eye(p)
    else:
        B = Bold = init
        
    active_itr = 1
    active_set_condition = True
    active_set = _getActiveSet( np.ones((p,p)) ) 

    while (active_itr==1) or (active_set_condition and active_itr <= max_active_itr):
        #print(f'active_itr: {active_itr}')

        for blocknum in prange(num_blocks+1):
            if blocknum == 0:
                B[0:partition[blocknum]][0:partition[blocknum]] = _blockPartialActive(blocknum, B, S, l, partition, p, eps, maxitr, active_set)[0:partition[blocknum]][0:partition[blocknum]]
            elif blocknum > 0 and blocknum < num_blocks:
                B[partition[blocknum-1]:partition[blocknum]][partition[blocknum-1]:partition[blocknum]] = _blockPartialActive(blocknum, B, S, l, partition, p, eps, maxitr, active_set)[partition[blocknum-1]:partition[blocknum]][partition[blocknum-1]:partition[blocknum]]
            elif blocknum == num_blocks:
                B[partition[blocknum-1]:p][partition[blocknum-1]:p] = _blockPartialActive(blocknum, B, S, l, partition, p, eps, maxitr, active_set)[partition[blocknum-1]:p][partition[blocknum-1]:p]
                
        new_active_set = _getActiveSet(B)

        if new_active_set.shape[0] == active_set.shape[0] and new_active_set.shape[1] == active_set.shape[1]:
            if (new_active_set == active_set).all():
                #print('Setting active set to False')
                active_set_condition = False

        active_set = new_active_set 

        active_itr += 1
    
    return B

    

class PDAG:
    def __init__(self, partitions, non_active=False):
        self.partitions = partitions
        self.Bhat = None
        self.non_active = non_active


    def fit(self, X, l, partitions, eps = 10^(-4), max_itr = 10, max_active_itr = 4, init=None):

        X = standardize(X)
        
        for partition in partitions:
            assert isinstance(partition, int), f'{partition} must be an integer'
        partitions = np.array(partitions)

        if self.non_active:
            self.Bhat = partial(X, l, partitions, eps = eps, maxitr = max_itr, init = init)
        else:
            self.Bhat = partialActive(X, l, partitions, eps = eps, maxitr = max_itr, max_active_itr = max_active_itr, init = init)

        return self.Bhat

    def __oldfit(self,X, l, m1=None, m2=None, m3=None, eps = 10^(-4), max_itr = 10, max_active_itr = 4, init=None):
        """
        this is deprecated
        """
        X = standardize(X)
        if self.partitions == 2:
            assert isinstance(m1, int), 'm1 must be an integer'
            self.Bhat = partial2(X, l, m1, eps, max_itr, init)
        elif self.partitions == 3:
            assert isinstance(m1, int) and isinstance(m2, int), 'm1 and m2 must be integers'
            self.Bhat = partial3(X, l, m1, m2, eps, max_itr, init)
        elif self.partitions == 4:
            assert isinstance(m1, int) and isinstance(m2, int) and isinstance(m3, int), 'm1, m2 and m3 must be integers'
            if not self.non_active:
                self.Bhat = partial4(X, l, m1, m2, m3, eps, max_itr, max_active_itr, init)
            else:
                self.Bhat = partial4NA(X, l, m1, m2, m3, eps, max_itr, init)
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
