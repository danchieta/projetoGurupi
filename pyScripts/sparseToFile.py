import numpy as np
from scipy import sparse

def sparseSave(A, filename):
	# save sparse csc_matrix to a npz file
	np.savez(filename, data = A.data, shape = A.shape, indices = A.indices, indptr = A.indptr)


def sparseLoad(filename):
	# load sparse matrix 
	return sparse.csc_matrix((matfile['data'], matfile['indices'], matfile['indptr']), shape = matfile['shape'])

import srModel
import sys

invZ, _ = srModel.priorCovMat((100,150), savetoDisk = True)

print 'size of dense matrix:', sys.getsizeof(invZ), 'bytes'

Z = sparse.csc_matrix(invZ)

sparseSave(Z, 'sparseZ.npz')