import numpy as np
from scipy.sparse import csr_matrix

def save_npz(file, matrix, vocab, compressed=False):
    arrays_dict = {}
    arrays_dict.update(indices=matrix.indices, indptr=matrix.indptr)
    arrays_dict.update(
        shape=matrix.shape,
        data=matrix.data,
        vocab=vocab
    )
    if compressed:
        np.savez_compressed(file, **arrays_dict)
    else:
        np.savez(file, **arrays_dict)

def load_npz(file):
    with np.load(file) as loaded:
        matrix = csr_matrix((loaded['data'], loaded['indices'], loaded['indptr']), shape=loaded['shape'])
        vocab = loaded['vocab']
        return (matrix,vocab)
