from scipy.sparse import linalg
from scipy.sparse import csr_matrix

print linalg.norm(csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]]), ord = 2)
