#   -*- coding: utf-8 -*-
# @Author  : Xiaoya Liu
# @Time    : 2023/4/7 11:48
# @File    : test.py

import planner.planning_utils

def main():
    # x = [1, 2, 3, 4, 5]
    # for i in range(len(x)-1, -1, -1):
    #     print(x[i])

    from cvxopt import solvers, matrix

    P = matrix([[2., 1.], [1., 2.]])
    q = matrix([2., 1.])
    G = matrix([[-1., 0.], [0., -1.]])
    h = matrix([1., 1.])
    A = matrix([1., 1.], (1, 2))
    b = matrix(1.)

    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h, A, b)

    print(sol)
    print(sol['x'])
    print(sol['primal objective'])


if __name__ == "__main__":
    main()
