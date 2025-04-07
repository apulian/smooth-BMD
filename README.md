This function numerically computes a smooth joint-minimum-variation Bloch–Messiah decomposition (see reference below) of a 2n-by-2n conjugate symplectic matrix-valued function FUN of one real parameter.

The singular values larger than 1 are arranged in decreasing order. The remaining singular values are organized such that the resulting diagonal matrix of singular values is symplectic. It is assumed that all singular values are distinct for every value of the parameter.

A typical call to smooth_BMD is:

Tout, Uout, Dout, Vout, flag = smooth_bmd(matfun, tspan)

See the script example_smooth_bmd.py for a usage example.

If you use this software, please cite the reference below.

Reference:
L. Dieci and A. Pugliese, "SVD, joint-MVD, Berry phase, and generic loss of rank for a matrix valued function of 2 parameters," Linear Algebra and its Applications, Volume 700, 2024, Pages 137-157, DOI: https://doi.org/10.1016/j.laa.2024.07.021

Authors: Giuseppe Patera and Alessandro Pugliese
