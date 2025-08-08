import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def test_cosine_similarity_shape():
    mat = np.eye(5)
    sim = cosine_similarity(mat)
    assert sim.shape == (5, 5)

def test_svd_decomposition():
    mat = np.random.rand(5, 5)
    u, s, vt = np.linalg.svd(mat, full_matrices=False)
    assert u.shape[1] == len(s) == vt.shape[0]
