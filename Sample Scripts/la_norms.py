import numpy as np

x = np.array([2,3,4])

l1 = np.linalg.norm(x,ord=1) ##l1 form


l2 = np.linalg.norm(x) ##default in l2 form

linf = np.linalg.norm(x,ord=np.inf) ##infinity norm

print("l1 norm:",l1)

print("l2 norm:",l2)

print("infiinity norm:",linf)

