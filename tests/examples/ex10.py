# coding: utf-8

from numpy import zeros

w_u = zeros(shape=4, dtype=float)
w_v = zeros(shape=4, dtype=float)

b_0 = zeros(shape=(4,4), dtype=float)
b_s = zeros(shape=(4,4), dtype=float)

for i_u in range(0,4):
    for i_v in range(0,4):
        for j_u in range(0,4):
            for j_v in range(0,4):
                contribution = 0
                for q_u in range(0,4):
                    for q_v in range(0,4):
                        ni_u = b_s[i_u,q_u] * b_0[i_v,q_v]
                        ni_v = b_0[i_u,q_u] * b_s[i_v,q_v]
                        nj_u = b_s[j_u,q_u] * b_0[j_v,q_v]
                        nj_v = b_0[j_u,q_u] * b_s[j_v,q_v]
                        r = ni_u*nj_u + ni_v*nj_v
                        contribution = contribution + r
