import numpy as np
from itertools import product

## FUNCTIONS ##

def find_cols(matrix):
    """
    Finds the columns of the given binary matrix that form an identity matrix.
    """
    nrows, _ = matrix.shape
    identity_matrix = np.eye(nrows, dtype=int)
    identity_indices = []

    for identity_col in identity_matrix.T:
        for col_index in range(matrix.shape[1]):
            if np.array_equal(matrix[:, col_index], identity_col):
                identity_indices.append(col_index)
                break
        if len(identity_indices) == nrows:
            break

    return identity_indices

def permute(H):
    """
    Finds H_hat
    """
    perm_indices = find_cols(H)
    nrows, ncols = H.shape
    swapped_columns = {}

    for index, perm_index in enumerate(perm_indices):
        target_index = ncols - nrows + index
        if perm_index in swapped_columns and swapped_columns[perm_index] == target_index:
            continue

        H[:, [perm_index, target_index]] = H[:, [target_index, perm_index]]

        swapped_columns[perm_index] = target_index
        swapped_columns[target_index] = perm_index

    return H

def check(H_hat, G):
    """
    Validate if H_hat * G * t = 0 for all t in F2.
    """

    k = G.shape[1]

    all_t = [np.array(t, dtype=int) for t in product([0, 1], repeat=k)]

    for t in all_t:
        t.reshape(-1,1)
        codeword = (G @ t) % 2
        if not np.all((H_hat @ codeword) % 2 == 0):
            return False  ## Condition failed for some t ##

    return True  ## Condition satisfied for all t ##


def compute(H):
    """
    Constructs the generator matrix G from the echelon form of H = [P | I_{n-k}].
    """
    H = permute(H)
    nk, n= H.shape
    k = n - nk  ## Dimension of the code

    P = H[:, :k]

    ## Construct G as [I_k | P]^T ##
    I_k = np.eye(k, dtype=int)

    G = np.vstack((I_k,P))

    return G, H


H = np.array([
    [1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 1],
    [1, 0, 0, 1, 1, 0]], dtype=int)


G, H_hat = compute(H)

print("H:")
print(H)
print("H_hat:")
print(H_hat)
print("\nG:")
print(G)

# print(check(H_hat,G))


