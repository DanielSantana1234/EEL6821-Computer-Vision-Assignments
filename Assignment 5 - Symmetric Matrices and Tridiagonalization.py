"""
V. Symmetric Matrices and Tridiagonalization
Using co-variance matrices, which are symmetric,
Tri-diagonalize the covariance matrix using (1) Householder method, 
(2) Givens Method and 
(3) the Jacobi method. Keep track of the number of iterations required for each method
Verify that values in the diagonal of the final Jacobi matrix are indeed the eigenvalues of the covariance matrix.
Are all the eigenvalues similar for all three methods?
Determine the total processing time for each method, and plot the required processing time 
versus dimension N of the N x N matrix
Check for all three methods if the eigenvalues obtained are the same as those of the 
original covariance matrix, then see if the eigenvectors of the respective tri-diagonalized matrices 
are related to those of the covariance matrix by the relation yi=T-1xi
"""