# coding: utf-8

from plaf.matrix_csr import plf_t_matrix_csr

n_rows = 8
n_cols = 8
n_nnz  = 18
n_block_rows = 1
n_block_cols = 1

n_global_rows = n_rows * n_block_rows
n_global_cols = n_cols * n_block_cols

M = plf_t_matrix_csr(n_rows, n_cols, n_nnz, n_block_rows, n_block_cols)


del M
