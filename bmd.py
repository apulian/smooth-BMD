import numpy as np

def bmd(A, q):
    """
    This function computes a Bloch-Messiah decomposition of a 2n-by-2n 
    conjugate symplectic matrix A. It returns the q largest singular 
    values and associated vectors.
    """
    
    U, s, Vh = np.linalg.svd(A)
    V = Vh.T.conj()
    m = A.shape[0]
    n = m // 2
    c = list(range(n)) + list(range(2 * n - 1, n - 1, -1))
    s = s[c]
    U = U[:, c]
    V = V[:, c]

    U11, U12 = U[:n, :n], U[:n, n:]
    U21, U22 = U[n:, :n], U[n:, n:]
    r_diag = np.diag(np.vstack((U22, U12)).T.conj() @ np.vstack((U11, -U21)))

    if np.isrealobj(r_diag):
        THETA = np.diag(np.concatenate((np.ones(n), np.sign(r_diag))))
    else:
        a = np.angle(r_diag) / 2
        THETA = np.diag(np.exp(np.concatenate((-1j * a, 1j * a))))

    U = U @ THETA[:, :2 * n]
    V = V @ THETA[:, :2 * n]
    return U[:, :q], s[:q], V[:, :q]