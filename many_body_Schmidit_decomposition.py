# https://www.zhihu.com/question/430091665/answer/1573803275
import torch
from math import log2, sqrt
'''
input:  wavefunction   => list [complex amplitudes]
        selected spins => list [qk...]
output: spectrum of entanglement
        Schmidt rank
'''

def bipartite(wavefunction, A_indices):
    # System of spins: [q0, q1, ..., qn]
    Hilbert_dims = len(wavefunction)
    sys_num = int( log2(Hilbert_dims) )
    sys_shape = (2,)*sys_num
    wf_tensor = wavefunction.view(size=sys_shape) # transform state vector to tensor.

    # Group part-A and part-B
    n_A = len(A_indices)
    n_B = sys_num - n_A
    A_new_indices = list(range(n_A)) # Selected spins belong to part-A on the left.

    # Permute indices of the system coefficent tensor and build new coefficent tensor.
    # also reference -> https://www.tensors.net/p-tutorial-1
    new_tensor = wf_tensor.moveaxis(A_indices, A_new_indices)
    new_matrix = new_tensor.view(2**n_A, 2**n_B)

    # SVD decomposition
    u, s, v = torch.svd(new_matrix)

    # Von Neumann entropy, log = ln
    p = s**2
    S_AB = -1 * (p * torch.log(p)).sum()

    return S_AB

Bell_state = torch.tensor([0, 1/sqrt(2), 1/sqrt(2), 0])
single_1 = torch.tensor([1j/sqrt(3), sqrt(2/3)])
single_2 = torch.tensor([1/2, sqrt(3)/2])
double_1 = torch.tensor([1/2, 1/2, 1/2, 1/2])
wf = torch.kron(double_1, Bell_state)
print( bipartite(wf, [0, 2]) )
