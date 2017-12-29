# coding: utf-8

a = lambda u,v: dx(u) * dx(v) + dy(u) * dy(v)

## vesion 1
#glt = glt_symbol(a)

# vesion 2
glt = eval('glt_symbol', a)
