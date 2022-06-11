# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
""" Module containing functions for testing the Bellman-Ford algorithm using pyccel or pythran
"""

# ================================================================
def bellman_ford ( v_num: int, e_num: int, source: int, e: 'int[:,:]', e_weight: 'real[:]',
                   v_weight: 'real[:]', predecessor: 'int[:]' ):
    """ Calculate the shortest paths from a source vertex to all other
    vertices in the weighted digraph
    """

    r8_big = 1.0E+14

    #  Step 1: initialize the graph.
    for i in range ( 0, v_num ):
        v_weight[i] = r8_big
    v_weight[source] = 0.0

    predecessor[:v_num] = -1

    #  Step 2: Relax edges repeatedly.
    for i in range ( 1, v_num ):
        for j in range ( e_num ):
            u = e[1][j]
            v = e[0][j]
            t = v_weight[u] + e_weight[j]
            if ( t < v_weight[v] ):
                v_weight[v] = t
                predecessor[v] = u

    #  Step 3: check for negative-weight cycles
    for j in range ( e_num ):
        u = e[1][j]
        v = e[0][j]
        if ( v_weight[u] + e_weight[j] < v_weight[v] ):
            # TODO BUG
            #print ( '' )
            print ( 'BELLMAN_FORD - Fatal error!' )
            print ( '  Graph contains a cycle with negative weight.' )
            return 1

    return 0

# ================================================================
# pythran export bellman_ford_test()
#@stack_array('e','e_weight','v_weight','predecessor')
def bellman_ford_test ( ):
    """ Test bellman ford's algorithm
    """

    from numpy import array
    from numpy import zeros

    e_num = 10
    v_num = 6

    e = array( (( 1, 4, 1, 2, 4, 2, 5, 3, 5, 3 ), \
                ( 0, 1, 2, 4, 0, 5, 0, 2, 3, 0 )) )

    e_weight = array( (-3.0,  6.0, -4.0, -1.0,  4.0, \
                       -2.0,  2.0,  8.0, -3.0,  3.0 ) )

    source = 0

    v_weight = zeros ( 6, dtype = float )
    predecessor = zeros ( 6, dtype = int )

    return bellman_ford ( v_num, e_num, source, e, e_weight, v_weight, predecessor )
