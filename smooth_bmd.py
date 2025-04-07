import numpy as np
import warnings
from bmd import bmd

def smooth_bmd(FUN, tspan, params=None):
    """
    smooth_bmd   Continuation of a smooth joint-minimum-variation
    Bloch-Messiah decomposition of a conjugate symplectic matrix-valued
    function of one parameter. It is possible to continue a subset of
    singular values and (corresponding) singular vectors.

    This function numerically computes a smooth joint-minimum-variation
    Bloch-Messiah decomposition of a 2n-by-2n conjugate symplectic
    matrix-valued function FUN of one real parameter.

    Parameters
    ----------
    FUN : callable
        A function that returns a 2n-by-2n conjugate symplectic matrix for a given scalar input.
    tspan : array-like of shape (2,)
        Start and end values of the continuation parameter.
    params : dict, optional
        Dictionary of optional parameters:
            - 'q': number of largest singular values to follow (default: n)
            - 'hmin': minimum step size (default: 1e-14)
            - 'hmax': maximum step size (default: abs(diff(tspan)) * 0.1)
            - 'tol': tolerance for variation (default: 1e-2)
            - 'h0': initial step size (default: min(hmax, tol) / 8)
            - 'maxstep': max number of steps (default: 1e4)
            - 'BMD0': initial BMD dict with keys 'U', 'D', 'V' (default: None)
            - 'fulloutput': bool (default: True)

    Returns
    -------
    Tout : list of float
        Parameter values at which BMD was computed.
    Uout : ndarray
        3D array of shape (2n, q, len(Tout)) with left singular vectors.
    Dout : ndarray
        2D array of shape (q, len(Tout)) with singular values.
    Vout : ndarray
        3D array of shape (2n, q, len(Tout)) with right singular vectors.
    flag : int
        Exit flag: 0 = success, -1 = min step reached, -2 = max steps reached.
    """
    
    t0, tfin = tspan
    A = FUN(t0)
    n = A.shape[0]

    if n % 2 != 0:
        raise ValueError("Input matrix function must be of even dimension (2n-by-2n).")

    tol_fctr = 1.1
    q = n
    h0_not_given = True
    hmin = 1e-14
    hmax = abs(tfin - t0) * 0.1
    maxstep = int(1e4)
    tol = 1e-2
    fulloutput = True
    BMD0 = None

    if params:
        q = params.get('q', q)
        h = params.get('h0', None)
        h0_not_given = h is None
        hmin = params.get('hmin', hmin)
        hmax = params.get('hmax', hmax)
        maxstep = params.get('maxstep', maxstep)
        tol = params.get('tol', tol)
        BMD0 = params.get('BMD0', None)
        fulloutput = params.get('fulloutput', fulloutput)

    tol /= tol_fctr
    if h0_not_given:
        h = min(hmax, tol) / 8

    try:
        _ = np.zeros((2 * n + 1, q, maxstep + 1), dtype=complex)
    except MemoryError:
        maxstep = round(0.7 * maxstep)
        warnings.warn(f"Maximum number of steps reduced to {maxstep} to avoid exceeding memory")

    if BMD0 is None:
        U, D, V = bmd(A, q)
    else:
        U, D, V = BMD0['U'], BMD0['D'], BMD0['V']

    Tout = [t0]
    Dout = [D]
    Uout = [U]
    Vout = [V]
    laststep = False
    s = np.sign(tfin - t0)
    h *= s

    for _ in range(maxstep):
        t1 = t0 + h
        if s * (tfin - t1) < 0:
            t1 = tfin
            laststep = True

        U, D, V, t1, h, step_flag = one_step(q, FUN, t0, t1, U, D, V, hmin, hmax, tol, tol_fctr)
        if step_flag == 0:
            t0 = t1
            if fulloutput:
                Tout.append(t1)
                Dout.append(D)
                Uout.append(U)
                Vout.append(V)

        if (laststep and tfin == t1) or step_flag == -1:
            if not fulloutput:
                Tout.append(tfin)
                Dout.append(D)
                Uout.append(U)
                Vout.append(V)
            return Tout, np.stack(Uout, axis=2), np.column_stack(Dout), np.stack(Vout, axis=2), step_flag

    return Tout + [t1], np.stack(Uout + [U], axis=2), np.column_stack(Dout + [D]), np.stack(Vout + [V], axis=2), -2


def one_step(q, f, t0, t1, U0, D0, V0, hmin, hmax, tol, tol_fctr):
    """
    Attempts a single continuation step.
    """
    h = t1 - t0
    s = np.sign(h)
    flag = -1
    failed = False

    while flag == -1:
        A = f(t1)
        flag = 0
        U1, D1, V1 = bmd(A, q)

        # Solve Procrustes problem
        PHI = np.diag(np.exp(1j * np.angle(np.diag(U1.conj().T @ U0 + V1.conj().T @ V0))))
        U1 = U1 @ PHI
        V1 = V1 @ PHI

        varD = D1 - D0
        varU = np.linalg.norm(U1 - U0, 'fro') / np.sqrt(q)
        varV = np.linalg.norm(V1 - V0, 'fro') / np.sqrt(q)

        if np.all(varD == 0):
            nwD, hpD = 1, 2 * h
        else:
            nwD = np.linalg.norm(varD / (tol * np.abs(D1) + tol), ord=np.inf)
            hpD = h / nwD

        nwU = 1 if varU == 0 else varU / tol
        hpU = 2 * h if varU == 0 else h / nwU

        nwV = 1 if varV == 0 else varV / tol
        hpV = 2 * h if varV == 0 else h / nwV

        if max(nwD, nwU, nwV) > tol_fctr:
            flag = -1

        hpred = np.abs([hpD, hpU, hpV, hmax, 2 * abs(h)])
        h = s * min(hpred)

        if flag == 0:
            if failed and abs(h) > abs(h0):
                h = h0
        else:
            failed = True
            h0 = h
            if abs(h) < hmin:
                return U0, D0, V0, t1, h, -1
            else:
                t1 = t0 + h

    return U1, D1, V1, t1, h, 0
