# coding: utf-8

from plaf.matrix_dns import plf_t_matrix_dns

n_rows = 4
n_cols = 4
n_block_rows = 1
n_block_cols = 1

n_global_rows = n_rows * n_block_rows
n_global_cols = n_cols * n_block_cols

a = zeros((n_global_rows, n_global_cols), double)

M = plf_t_matrix_dns(n_rows, n_cols, n_block_rows, n_block_cols)

a[1,1] = 1.0

n = M.n_rows
n = 2*M.n_rows + 1

M.n_rows = n + 1 - 1

M.arr_a = a + 1.0
x = 0.5
#M.arr_a[1,1] = 2.0*x + 1.0


del M
