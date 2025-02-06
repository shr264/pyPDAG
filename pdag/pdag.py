from scipy.stats import chi2
import numpy as np
import pandas as pd
import networkx as nx
from numba import njit, prange
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt



# --------------------------- EM Imputation (Placeholder) ---------------------------
def simple_em_impute_data(df, max_iter=20, tol=1e-5):
    """
    Very simplified EM-like approach for completing missing data in a DataFrame df,
    assuming (roughly) a multivariate Gaussian structure.
    In a real application, you might use 'fancyimpute' or 'sklearn.impute.IterativeImputer'
    or a more specialized EM routine.

    Returns
    -------
    df_filled : pd.DataFrame with no missing values (NaNs filled).
    """
    df_filled = df.copy()
    n, p = df_filled.shape

    # 1) Initialize missing entries with column means
    for col in df_filled.columns:
        if df_filled[col].isna().any():
            mean_val = df_filled[col].mean(skipna=True)
            df_filled[col].fillna(mean_val, inplace=True)

    # 2) Iteratively update via conditional means
    for _ in range(max_iter):
        old_df = df_filled.copy()

        # M-step: estimate mean, cov from the completed data
        mu = df_filled.mean(axis=0).values  # shape (p,)
        Sigma = df_filled.cov().values      # shape (p,p)

        # E-step (approx): for each row w/ missing in original df
        for i in range(n):
            row_mask = df.iloc[i].isna()
            if row_mask.any():
                # Split indices into observed vs missing
                obs_idx = np.where(~row_mask)[0]
                mis_idx = np.where(row_mask)[0]

                # Partition the mean/cov
                mu_obs = mu[obs_idx]
                mu_mis = mu[mis_idx]
                Sigma_oo = Sigma[np.ix_(obs_idx, obs_idx)]
                Sigma_mm = Sigma[np.ix_(mis_idx, mis_idx)]
                Sigma_mo = Sigma[np.ix_(mis_idx, obs_idx)]
                Sigma_om = Sigma[np.ix_(obs_idx, mis_idx)]

                x_obs = df_filled.iloc[i, obs_idx].values
                # conditional mean = mu_mis + Sigma_mo * Sigma_oo^{-1} * (x_obs - mu_obs)
                try:
                    Sigma_oo_inv = np.linalg.inv(Sigma_oo)
                except np.linalg.LinAlgError:
                    # fallback: just keep old values
                    continue

                cond_mean = mu_mis + Sigma_mo.dot(Sigma_oo_inv).dot(x_obs - mu_obs)
                # Update the row in df_filled
                for idx_m, val_m in zip(mis_idx, cond_mean):
                    df_filled.iat[i, idx_m] = val_m

        # Check for convergence
        diff = (df_filled - old_df).abs().max().max()
        if diff < tol:
            break

    return df_filled


def _get_partitioned_positions(p, partitions):
    """
    p           : total number of nodes
    partitions  : list of partition boundaries, e.g. [m1, m2, ...]
                  (same as self.partitions in your code)
    Returns:
      A dict: { node_index: (x_coord, y_coord) }
      so that each partition's nodes are placed on a distinct horizontal row.
    """
    # Breakpoints: [0, m1, m2, ..., p]
    breakpoints = [0] + list(partitions) + [p]

    pos = {}
    # For each partition index i, we have an interval [start, end)
    for i in range(len(breakpoints) - 1):
        start = breakpoints[i]
        end = breakpoints[i + 1]
        # The row index = -i
        # We'll assign x = 0..(end-start-1)
        for offset, node in enumerate(range(start, end)):
            pos[node] = (offset, -i)

    return pos


# --------------------------- Numba-compiled PDE/PDAG logic -------------------------


@njit(fastmath=True)
def _softhresh(x, l):
    return np.sign(x) * max(abs(x) - l, 0)


@njit(fastmath=True)
def _getActiveSet(B):
    nonzeros = np.nonzero(B)
    return np.column_stack((nonzeros[0], nonzeros[1]))


@njit(fastmath=True)
def _boundBii(out):
    return min(abs(out), 1.0)


@njit(fastmath=True)
def _Bii(S, B, i):
    p = len(B)
    sumterm = 0.0
    for k in range(p):
        if k != i:
            sumterm += S[i, k] * B[i, k]
    tmp = sumterm * sumterm
    out = (-sumterm + np.sqrt(tmp + 4.0 * S[i, i])) / (2.0 * S[i, i])
    return _boundBii(out)


@njit(fastmath=True)
def _boundBik(out):
    if out > 0:
        return min(out, 1.0)
    else:
        return max(out, -1.0)


@njit(fastmath=True)
def _Bik(S, B, i, j, l):
    sumterm = 0.0
    p = B.shape[0]
    for k in range(p):
        if k != j:
            sumterm += S[j, k] * B[i, k]
    out = -sumterm / (2.0 * S[j, j])
    out = _softhresh(out, l / (4.0 * S[j, j]))
    if out != 0.0:
        out = _boundBik(out)
    return out


@njit(fastmath=True)
def _isCyclicUtil(v, visited, recStack, graph, p, m1, m2):
    visited[v] = True
    recStack[v] = True
    for neighbor in range(m1, m2):
        if neighbor != v and graph[v, neighbor] == 1:
            if not visited[neighbor]:
                if _isCyclicUtil(neighbor, visited, recStack, graph, p, m1, m2):
                    return True
            elif recStack[neighbor]:
                return True
    recStack[v] = False
    return False


@njit(fastmath=True)
def _isCyclic(B, i, j, m1, m2):
    graph = (B != 0).astype(np.uint8)
    graph[i, j] = 1
    p = len(B)
    visited = np.zeros(p, dtype=np.bool_)
    recStack = np.zeros(p, dtype=np.bool_)
    if not visited[j]:
        if _isCyclicUtil(j, visited, recStack, graph, p, m1, m2):
            return True
    return False


@njit(fastmath=True)
def _qBij(S, Bij, B, i, j, l):
    p = len(S)
    sumterm = 0.0
    for k in range(p):
        if k != j:
            sumterm += S[j, k] * B[i, k]
    out1 = (S[j, j] * Bij * Bij) + (2.0 * Bij * sumterm) + (l * abs(Bij))
    out2 = 0.0
    if out1 < out2:
        return out1, Bij
    else:
        return 0.0, 0.0


@njit(fastmath=True)
def _diagonalBlock(B, S, l, m1, m2, p):
    for i in range(m1, m2):
        B[i, i] = _Bii(S, B, i)
    for i in range(m1 + 1, m2):
        for k in range(m1, i):
            if _isCyclic(B, i, k, m1, m2):
                B[i, k] = 0.0
                B[k, i] = _Bik(S, B, k, i, l)
            elif _isCyclic(B, k, i, m1, m2):
                B[k, i] = 0.0
                B[i, k] = _Bik(S, B, i, k, l)
            else:
                Bik = _Bik(S, B, i, k, l)
                Bki = _Bik(S, B, k, i, l)
                qbki = _qBij(S, Bki, B, k, i, l)
                qbik = _qBij(S, Bik, B, i, k, l)
                if qbki[0] > qbik[0]:
                    B[i, k] = qbik[1]
                    B[k, i] = 0.0
                elif qbki[0] < qbik[0]:
                    B[k, i] = qbki[1]
                    B[i, k] = 0.0
                else:
                    u = np.random.rand()
                    if u < 0.5:
                        B[i, k] = qbik[1]
                        B[k, i] = 0.0
                    else:
                        B[k, i] = qbki[1]
                        B[i, k] = 0.0
    return B


@njit(fastmath=True)
def _diagonalBlockActive(B, S, l, m1, m2, p, active_set):
    for i in range(m1, m2):
        B[i, i] = _Bii(S, B, i)
    for (i, k) in active_set:
        if (m1 + 1) <= i < m2 and m1 <= k < i:
            if _isCyclic(B, i, k, m1, m2):
                B[i, k] = 0.0
                B[k, i] = _Bik(S, B, k, i, l)
            elif _isCyclic(B, k, i, m1, m2):
                B[k, i] = 0.0
                B[i, k] = _Bik(S, B, i, k, l)
            else:
                Bik = _Bik(S, B, i, k, l)
                Bki = _Bik(S, B, k, i, l)
                qbki = _qBij(S, Bki, B, k, i, l)
                qbik = _qBij(S, Bik, B, i, k, l)
                if qbki[0] > qbik[0]:
                    B[i, k] = qbik[1]
                    B[k, i] = 0.0
                elif qbki[0] < qbik[0]:
                    B[k, i] = qbki[1]
                    B[i, k] = 0.0
                else:
                    u = np.random.rand()
                    if u < 0.5:
                        B[i, k] = qbik[1]
                        B[k, i] = 0.0
                    else:
                        B[k, i] = qbki[1]
                        B[i, k] = 0.0
    return B


@njit(fastmath=True)
def _offDiagBlock(B, S, l, m1, m2, p):
    for i in range(m1 + 1, m2):
        for k in range(m1):
            B[k, i] = 0.0
            B[i, k] = _Bik(S, B, i, k, l)
    return B


@njit(fastmath=True)
def _offDiagBlockActive(B, S, l, m1, m2, p, active_set):
    for (i, k) in active_set:
        if (m1 + 1) <= i < m2 and 0 <= k < m2:
            B[k, i] = 0.0
            B[i, k] = _Bik(S, B, i, k, l)
    return B


@njit(fastmath=True)
def _blockPartial(blocknum, B, S, l, partition, p, eps, maxitr):
    Bold = B.copy()
    diff = 1.0
    itr = 1
    num_blocks = len(partition)

    start_idx = 0 if blocknum == 0 else partition[blocknum - 1]
    end_idx = partition[blocknum] if blocknum < num_blocks else p

    while diff > eps and itr <= maxitr:
        if blocknum == 0:
            B = _diagonalBlock(B, S, l, 0, end_idx, p)
        else:
            B = _offDiagBlock(B, S, l, start_idx, end_idx, p)
            B = _diagonalBlock(B, S, l, start_idx, end_idx, p)
        diff = np.max(np.abs(B - Bold))
        Bold = B.copy()
        itr += 1
    return B


@njit(fastmath=True)
def _blockPartialActive(blocknum, B, S, l, partition, p, eps, maxitr, active_set):
    Bold = B.copy()
    diff = 1.0
    itr = 1
    num_blocks = len(partition)

    start_idx = 0 if blocknum == 0 else partition[blocknum - 1]
    end_idx = partition[blocknum] if blocknum < num_blocks else p

    while diff > eps and itr <= maxitr:
        if blocknum == 0:
            B = _diagonalBlockActive(B, S, l, 0, end_idx, p, active_set)
        else:
            B = _offDiagBlockActive(B, S, l, start_idx, end_idx, p, active_set)
            B = _diagonalBlockActive(B, S, l, start_idx, end_idx, p, active_set)
        diff = np.max(np.abs(B - Bold))
        Bold = B.copy()
        itr += 1
    return B


@njit(parallel=True)
def partial(X, l, partition, eps=1e-4, maxitr=10, init=None):
    n, p = X.shape
    S = np.dot(X.T, X) / n
    if init is None:
        B = np.eye(p)
    else:
        B = init.copy()

    num_blocks = len(partition)
    for blocknum in range(num_blocks + 1):
        B = _blockPartial(blocknum, B, S, l, partition, p, eps, maxitr)
    return B


@njit(parallel=True)
def partialActive(X, l, partition, eps=1e-4, maxitr=10, max_active_itr=10, init=None):
    n, p = X.shape
    S = np.dot(X.T, X) / n
    if init is None:
        B = np.eye(p)
    else:
        B = init.copy()

    num_blocks = len(partition)
    active_itr = 1
    active_set_condition = True

    active_set = _getActiveSet(np.ones((p, p)))  # start: everything active

    while (active_itr == 1) or (active_set_condition and active_itr <= max_active_itr):
        for blocknum in range(num_blocks + 1):
            B = _blockPartialActive(blocknum, B, S, l, partition, p, eps, maxitr, active_set)

        new_active_set = _getActiveSet(B)
        if new_active_set.shape[0] == active_set.shape[0]:
            if (new_active_set == active_set).all():
                active_set_condition = False
        active_set = new_active_set
        active_itr += 1

    return B

# ---------------------------- PDAG Class ----------------------------------


##############################################################################
# Robust EM (single multivariate Gaussian) from previous discussion
##############################################################################

def robust_gaussian_em_impute(
    X,
    max_iter=50,
    tol=1e-6,
    robust=False,
    alpha=0.95,
    reg=1e-8,
    verbose=False
):
    """
    EM to fill missing data under a single multivariate Gaussian model.
    Optionally uses robust weighting in the M-step.
    """
    if isinstance(X, pd.DataFrame):
        colnames = X.columns
        X = X.values
    else:
        colnames = None

    n, p = X.shape
    X_imputed = X.copy()

    # Initialize missing with column means
    col_means = np.nanmean(X_imputed, axis=0)
    for j in range(p):
        mask = np.isnan(X_imputed[:, j])
        X_imputed[mask, j] = col_means[j]

    mu = np.mean(X_imputed, axis=0)
    Sigma = np.cov(X_imputed, rowvar=False) + np.eye(p)*reg

    for it in range(max_iter):
        X_old = X_imputed.copy()
        # E-step
        for i in range(n):
            row_mask = np.isnan(X[i])
            if row_mask.any():
                obs_idx = np.where(~row_mask)[0]
                mis_idx = np.where(row_mask)[0]
                mu_obs = mu[obs_idx]
                mu_mis = mu[mis_idx]
                Sigma_oo = Sigma[np.ix_(obs_idx, obs_idx)]
                Sigma_mm = Sigma[np.ix_(mis_idx, mis_idx)]
                Sigma_mo = Sigma[np.ix_(mis_idx, obs_idx)]
                Sigma_om = Sigma[np.ix_(obs_idx, mis_idx)]
                x_obs = X_imputed[i, obs_idx]
                try:
                    Sigma_oo_inv = np.linalg.inv(Sigma_oo)
                except np.linalg.LinAlgError:
                    continue
                cond_mean = mu_mis + Sigma_mo @ Sigma_oo_inv @ (x_obs - mu_obs)
                X_imputed[i, mis_idx] = cond_mean

        # M-step
        if robust:
            # Weighted (robust) M-step
            mu = np.mean(X_imputed, axis=0)
            try:
                Sigma_inv = np.linalg.inv(Sigma)
            except np.linalg.LinAlgError:
                Sigma += np.eye(p)*reg
                Sigma_inv = np.linalg.inv(Sigma)
            dist_sq = np.zeros(n)
            for i in range(n):
                diff = X_imputed[i] - mu
                dist_sq[i] = diff @ Sigma_inv @ diff
            cutoff = chi2.ppf(alpha, df=p)
            wts = np.ones(n)
            for i in range(n):
                if dist_sq[i] > cutoff:
                    wts[i] = cutoff/dist_sq[i]
            w_sum = np.sum(wts)
            mu = np.sum(X_imputed*wts[:, None], axis=0)/w_sum
            Sigma_new = np.zeros((p, p))
            for i in range(n):
                diff = (X_imputed[i] - mu).reshape(-1, 1)
                Sigma_new += wts[i]*(diff@diff.T)
            Sigma = Sigma_new/w_sum + np.eye(p)*reg
        else:
            mu = np.mean(X_imputed, axis=0)
            Sigma = np.cov(X_imputed, rowvar=False) + np.eye(p)*reg

        diff = np.nanmax(np.abs(X_imputed - X_old))
        if verbose:
            print(f"Iteration {it+1}, diff={diff}")
        if diff < tol:
            break

    if colnames is not None:
        df_out = pd.DataFrame(X_imputed, columns=colnames)
        return df_out, mu, Sigma
    else:
        return X_imputed, mu, Sigma


##############################################################################
# Factor-correction code
##############################################################################
def correct_factor(df, factor_cols):
    """
    Factor-correct all numeric columns in 'df' using the factor columns (categorical).
    We do Y ~ C(factor1) + C(factor2) + ... and replace Y with residuals.
    """
    df_corrected = df.copy()
    if isinstance(factor_cols, str):
        factor_cols = [factor_cols]

    # Ensure factor cols are categorical
    for fc in factor_cols:
        df_corrected[fc] = df_corrected[fc].astype('category')

    # Identify numeric columns that are NOT factor columns
    numeric_cols = [
        c for c in df_corrected.columns
        if c not in factor_cols
        and pd.api.types.is_numeric_dtype(df_corrected[c])
    ]
    # Build formula
    factor_formula = " + ".join([f"C({fc})" for fc in factor_cols])

    for ycol in numeric_cols:
        formula = f"{ycol} ~ {factor_formula}"
        model = smf.ols(formula=formula, data=df_corrected).fit()
        df_corrected[ycol] = model.resid
    return df_corrected

##############################################################################
# PDAG class with updated set_data method
##############################################################################


class PDAG:
    def __init__(self,
                 partitions,
                 non_active=False,
                 X=None,
                 complete_data=False,
                 factor_cols=None,
                 max_em_iter=20,
                 em_tol=1e-5,
                 robust=True,
                 column_names=None):
        """
        partitions    : list of partition boundaries
        non_active    : whether to do normal block updates or active-set updates
        X             : optional data
        complete_data : if True, do robust EM to fill missing numeric data
        factor_cols   : list of factor/categorical columns
        max_em_iter   : max EM iterations
        em_tol        : EM convergence tolerance
        robust        : whether M-step is robust or classical
        """
        self.partitions = partitions
        self.non_active = non_active
        self.X = None
        self.Bhat = None
        self.max_em_iter = max_em_iter
        self.em_tol = em_tol
        self.robust = robust
        self.complete_data = complete_data
        self.factor_cols = factor_cols
        self.column_names = column_names  

        if X is not None:
            self.set_data(X, complete_data, factor_cols)

    def set_data(self, X, complete_data, factor_cols):
        """
        Step 1: Convert X to DataFrame
        Step 2: If factor columns have missing data, drop them
        Step 3: Factor-correct numeric columns
        Step 4: If complete_data=True, run robust EM on numeric columns
        """

        if isinstance(X, pd.DataFrame):
            self.column_names = X.columns.tolist()

        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        # 1) Identify and drop factor cols that have missing data
        if factor_cols is not None:
            # Convert indices -> names if needed
            if all(isinstance(fc, int) for fc in factor_cols):
                factor_cols = [df.columns[fc] for fc in factor_cols]

            drop_list = []
            for fc in factor_cols:
                if df[fc].isna().any():
                    drop_list.append(fc)

            # Drop factor columns with missing data
            if drop_list:
                print("Dropping factor columns with missing data:", drop_list)
                factor_cols = [fc for fc in factor_cols if fc not in drop_list]
                df = df.drop(columns=drop_list)

            # 2) Factor correction on numeric columns
            if factor_cols:
                df = correct_factor(df, factor_cols)
            else:
                print("No valid factor columns remain for correction.")
        else:
            # No factor columns provided
            factor_cols = []

        # 3) If complete_data => robust EM on numeric columns
        if complete_data:
            # We only want to impute numeric columns;
            # any leftover non-numeric columns won't be touched
            # but let's see if they have missing data => we'd have to drop or error
            # For simplicity, let's just do full data
            # The robust_gaussian_em_impute returns a DataFrame if input is DataFrame
            df_filled, mu, Sigma = robust_gaussian_em_impute(
                df,
                max_iter=self.max_em_iter,
                tol=self.em_tol,
                robust=self.robust,
                verbose=False
            )
            self.X = df_filled
        else:
            # If user did not want to complete data, check if there's still any missing
            if df.isna().any().any():
                raise ValueError("X has missing data, but complete_data=False. Stopping.")
            self.X = df

        

    def fit(self, l, partitions=None, eps=1e-4, max_itr=10, max_active_itr=4, init=None):
        if partitions is None:
            partitions = self.partitions
        if self.X is None:
            raise ValueError("No data. Call set_data(...) or provide X in constructor.")

        # Convert to NumPy
        df = self.X
        X_array = df.to_numpy().astype(float)

        # Standardize columns
        mu = X_array.mean(axis=0)
        sigma = X_array.std(axis=0, ddof=1)
        sigma[sigma < 1e-12] = 1e-12
        X_std = (X_array - mu)/sigma

        # PDE logic
        partitions_arr = np.array(partitions, dtype=np.int64)

        if self.non_active:
            B = partial(X_std, l, partitions_arr, eps=eps, maxitr=max_itr, init=init)
        else:
            B = partialActive(X_std, l, partitions_arr, eps=eps, maxitr=max_itr,
                              max_active_itr=max_active_itr, init=init)
        self.Bhat = B
        return B

    def getAdjacency(self):
        if self.Bhat is None:
            raise ValueError("Must call fit(...) first.")

        Bhat = self.Bhat.copy()
        np.fill_diagonal(Bhat, 0)  # Set diagonal to 0
        return (np.abs(Bhat) > 1e-5)  # Compare absolute values with 0.00001

    def getGraph(self):
        return nx.from_numpy_array(self.getAdjacency(), create_using=nx.DiGraph())

    def plotGraph(self, column_names=None, debug=False):
        """
        Plots the graph with partitioned node positions, adjusting figure size dynamically.

        Args:
        - column_names (list, optional): Custom labels for nodes.
        """
        if self.Bhat is None:
            raise ValueError("No adjacency matrix available. Run `fit()` first.")

        p = self.Bhat.shape[0]

        # Use provided column names, else use stored names, else default numeric labels
        node_labels = column_names if column_names else self.column_names
        if node_labels is None or len(node_labels) != p:
            node_labels = [f"X{i}" for i in range(p)]  # Default numeric labels

        # Create graph from adjacency matrix
        G = nx.from_numpy_array(self.getAdjacency().T, create_using=nx.DiGraph)

        # Generate partitioned positions using original node indices
        pos_original = _get_partitioned_positions(p, self.partitions)

        # Relabel nodes in graph
        mapping = {i: node_labels[i] for i in range(p)}
        G = nx.relabel_nodes(G, mapping)

        # Update positions to match new node labels
        pos = {node_labels[i]: pos_original[i] for i in range(p)}

        # Determine dynamic figure size
        max_nodes_per_partition = max(
            [self.partitions[i + 1] - self.partitions[i] for i in range(len(self.partitions) - 1)]
        )  # Max nodes in a row
        num_partitions = len(self.partitions) - 1  # Number of partitions

        fig_width = max(4, max_nodes_per_partition * 3)  # Scale width
        fig_height = max(4, num_partitions*2)  # Scale height

        if debug: 
            print(max_nodes_per_partition, num_partitions, fig_width, fig_height)
        # Plot the graph
        plt.figure(figsize=(fig_width, fig_height))
        nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue",
                font_size=10, edge_color="gray")
        plt.title("PDAG Structure with Partitioned Layout")
        plt.show()
