
import numpy as np

def closure(mat):
    mat = np.atleast_2d(mat)
    if np.any(mat < 0):
        raise ValueError("Cannot have negative proportions")
    if mat.ndim > 2:
        raise ValueError("Input matrix can only have two dimensions or less")
    if np.all(mat == 0, axis=1).sum() > 0:
        raise ValueError("Input matrix cannot have rows with all zeros")
    mat = mat / mat.sum(axis=1, keepdims=True)
    return mat.squeeze()

def inner(x, y):
    x = closure(x)
    y = closure(y)
    a, b = clr(x), clr(y)
    return a.dot(b.T)


def _gram_schmidt_basis(n):
    basis = np.zeros((n, n-1))
    for j in range(n-1):
        i = j + 1
        e = np.array([(1/i)]*i + [-1] +
                     [0]*(n-i-1))*np.sqrt(i/(i+1))
        basis[:, j] = e
    return basis.T

def clr(mat):
    mat = closure(mat)
    lmat = np.log(mat)
    gm = lmat.mean(axis=-1, keepdims=True)
    return (lmat - gm).squeeze()


def clr_inv(mat):
    return closure(np.exp(mat))
    

def ilr(mat, basis=None, check=True):
    
    mat = closure(mat)
    if basis is None:
        basis = clr_inv(_gram_schmidt_basis(mat.shape[-1]))
    else:
        if len(basis.shape) != 2:
            raise ValueError("Basis needs to be a 2D matrix, "
                             "not a %dD matrix." %
                             (len(basis.shape)))
        if check:
            _check_orthogonality(basis)

    return inner(mat, basis)