import numpy as np
import scipy.linalg as la
import code

def doublePassG(A,B,Binv,k,p):
    # Generate Gaussian random matrix
    dim = A.shape[0]
#    np.random.seed(0)
    Omega = np.random.normal(0,1,size=(dim,k+p))
    
    # First pass    
    Ybar = A.matmat(Omega)
    Y = Binv.matmat(Ybar)

    # orothogonalize with B inner products
    Q = PreCholQr_W_inner(Y,B)

    # Second pass
    AQ = A.matmat(Q)
    
    # Q*AQ = VdV^T
    T = np.dot(Q.T, AQ)
    d, V = la.eigh(T)

    # Sort by descending eigenvalues 
    sort_perm = d.argsort()
    sort_perm = sort_perm[::-1]
    d = d[sort_perm[0:k]]
    V = V[:, sort_perm[0:k]]

    # project eigenvalues
    U = np.dot(Q,V)

    return d,U


def singlePassG(A,B,Binv,k,p):
    # Generate Gaussian random matrix
    dim = A.shape[0]
#:    np.random.seed(0)
    Omega = np.random.normal(0,1,size=(dim,k+p))

    # First pass
    Ybar = A.matmat(Omega)
    Y = Binv.matmat(Ybar)
    
    # orothogonalize with B inner products
    Q = PreCholQr_W_inner(Y,B)
    
    BQ = B.matmat(Q)
    OBQ = np.dot(Omega.T,BQ)
    OYbar = np.dot(Omega.T,Ybar)
    M = la.solve(OBQ,OYbar)
    T = la.solve(OBQ,M.T).T
    
    d, V = la.eigh(T)

    # Sort by descending eigenvalues
    sort_perm = d.argsort()
    sort_perm = sort_perm[::-1]
    d = d[sort_perm[0:k]]
    V = V[:, sort_perm[0:k]]

    # project eigenvalues
    U = np.dot(Q,V)
    return d,U


def CholQR_W_inner(Y,W):
    Z = W.matmat(Y)
    C = np.dot(Y.T,Z)
    R = la.cholesky(C)
    Q = la.solve_triangular(R,Y.T,trans=1).T
    return Q

def PreCholQr_W_inner(Y,W):
    Z,_ = la.qr(Y,mode='economic')
    Q = CholQR_W_inner(Y,W)
    return Q
