from petsc4py import PETSc as Pet
import numpy as np
import scipy.sparse

A = np.array([[2,0,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,2]])
print A
print A.shape

A_csr = scipy.sparse.csr_matrix(A)

A_petsc = Pet.Mat().createAIJ(size=A_csr.shape,
                    csr = (A_csr.indptr, A_csr.indices, A_csr.data))

x,b = A_petsc.getVecs()
b.set(1)
print b.getArray()
# create linear solver
ksp = Pet.KSP()
ksp.create(Pet.COMM_WORLD)
ksp.setType('cg')
pc = ksp.getPC()
pc.setType(pc.Type.GAMG)

ksp.setOperators(A_petsc)
ksp.solve(b,x)

print x.getArray()

# and incomplete Cholesky
#ksp.getPC().setType('icc')
# obtain sol & rhs vectors
#x, b = A.createVecs()
#x.set(0)
#b.set(1)
# and next solve
#ksp.setOperators(A)
#ksp.setFromOptions()
#ksp.solve(b, x)