import numpy as np

def fvecs_read(filename: str, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv

def ivecs_read(filename: str):
    ''' 
    The groundtruth files contain, for each query, the identifiers (vector number, starting at 0) of its k nearest neighbors, ordered by increasing (squared euclidean) distance. 
    Therefore, the first element of each integer vector is the nearest neighbor identifier associated with the query. 
    '''
    a = np.fromfile(filename, dtype=np.int32)
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()
