from cmath import inf
import math
from re import A
from this import d
import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from CubicSpline import Spline2D
import trapezoid_vel_profile as profile


'''
parameters
'''
start = [0.0, 0.0]
mid =  [[0.2, 3.2], [1.3, 3.2], [1.0, 0.0], [3.2, 0.7], [2.1, 1.7], [3.1, 2.7]]
goal =  [3.1, 3.8]

velocity = [0.0, 1.0, 0.0, 2.0, 2.0, 2.0, 0.0]

frequency = 1000 # hz

self_inf = 1e+10

robot_size = 0.5

field = [[-0.55, -0.55], [3.95, 3.55]]
v_max = 3
a_max = 4
d_max = 6

l_gain = 1e+1
c_gain = 1e+1

param = []
for i in range(len(mid)+1):
    if i == 0:
        param.append(start[0]+(mid[i][0]-start[0])/2)
        param.append(start[1]+(mid[i][1]-start[1])/2)
    elif i == len(mid):
        param.append(mid[i-1][0]+(goal[0]-mid[i-1][0])/2)
        param.append(mid[i-1][1]+(goal[1]-mid[i-1][1])/2)
    else:
        param.append(mid[i-1][0]+(mid[i][0]-mid[i-1][0])/2)
        param.append(mid[i-1][1]+(mid[i][1]-mid[i-1][1])/2)


'''
functions
'''
def spline(x, y):
    num=100
    sp = Spline2D(x, y)
    l_x, l_y, l_k, length_list = [], [], [], []
    for i in range(len(x)-1):
        s = np.linspace(sp.s[i], sp.s[i+1], num)[:-1]
        r_x, r_y = [], []
        for i_s in s:
            ix, iy = sp.calc_position(i_s)
            r_x.append(ix)
            l_x.append(ix)
            r_y.append(iy)
            l_y.append(iy)
            l_k.append(sp.calc_curvature(i_s))
        length = np.concatenate([[0.0], np.cumsum([np.hypot(dx, dy) for dx, dy in zip(np.diff(r_x), np.diff(r_y))]).tolist()])
        length_list.append(length[-1])

    return l_x, l_y, l_k, length_list

def judgeCross(a, b, c, d):
    if (((a[0] - b[0]) * (c[1] - a[1]) + (a[1] - b[1]) * (a[0] - c[0])) * ((a[0] - b[0]) * (d[1] - a[1]) + (a[1] - b[1]) * (a[0] - d[0])) < 0):
        if (((c[0] - d[0]) * (a[1] - c[1]) + (c[1] - d[1]) * (c[0] - a[0])) * ((c[0] - d[0]) * (b[1] - c[1]) + (c[1] - d[1]) * (c[0] - b[0])) < 0):
            return True

def calcTrajectory(param):
    p = param
    x = [start[0]]
    y = [start[1]]
    x.append(p[0])
    y.append(p[1])

    for i in range(len(mid)):
        x.append(mid[i][0])
        y.append(mid[i][1])
        x.append(p[2*i+2])
        y.append(p[2*i+3])

    x.append(goal[0])
    y.append(goal[1])
    
    r_x, r_y, r_k, length = spline(x, y)

    return x, y, r_x, r_y, r_k, length

# 目的関数
def func(param):
    x, y, r_x, r_y, r_k, length = calcTrajectory(param)
    cost = np.sum(length)*l_gain + np.max(np.abs(r_k))*c_gain
    print(min(r_x))

    return cost

'''
制約条件式
'''
def min_cons(param):
    p = param
    for i in range(len(p)):
        if i % 2 == 0:
            p[i] - field[0][0]+robot_size*0.5
        else:
            p[i] - field[0][1]+robot_size*0.5
    return p

def max_cons(param):
    p = param
    for i in range(len(p)):
        if i % 2 == 0:
            field[1][0]-robot_size*0.5 - p[i]
        else:
            field[1][1]-robot_size*0.5 - p[i]
    return p

# 座標被り回避
def over_cons(param):
    x, y, r_x, r_y, r_k, length = calcTrajectory(param)
    if  np.min(r_x) < field[0][0]+robot_size*0.5 or np.min(r_y) < field[0][1]+robot_size*0.5 or np.max(r_x) > field[1][0]-robot_size*0.5 or np.max(r_y) > field[1][1]-robot_size*0.5:
        return -self_inf
    else:
        return 0


def main():
    # 制約条件式が0以上になるようにする
    # ineq 不等号, eq   等号
    cons = (
        {'type': 'ineq', 'fun': min_cons},
        {'type': 'ineq', 'fun': max_cons},
        {'type': 'ineq', 'fun': over_cons}
    )

    result = minimize(func, x0=param, constraints=cons, method="COBYLA")
    print(result)

    x, y, r_x, r_y, r_k, length = calcTrajectory(result.x)

    fig = plt.figure(figsize=(10,4))
    ax1 = plt.subplot2grid((1, 3), (0, 0))
    ax2 = plt.subplot2grid((1, 3), (0, 1))
    ax3 = plt.subplot2grid((1, 3), (0, 2))
    x_end = 0.0
    v_end = 0.0
    t_list = []
    stack_t = 0
    for i in range(len(velocity)):
        (Y, Yd, Ydd, t) = profile.plan(x_end, length[i*2]+length[i*2+1], v_end, velocity[i], v_max, a_max, d_max, frequency)
        x_end = 0
        v_end = velocity[i]
        t_list.append(t[-1])
        ax1.plot(t, Y, label=str(i)) # Pos
        ax2.plot(stack_t+t, Yd, label=str(i)) # Vel
        stack_t += t[-1]
    print('m : {}'.format(sum(length)))
    print('s : {}'.format(sum(t_list)))

    ax3.scatter(r_x, r_y, s=5, label="trajectory")
    ax3.scatter(x, y, s=30, label="point")
    ax3.axis('square')
    ax3.grid(which='major', linestyle='-')
    plt.show()

if __name__ == '__main__':
    main()

# 座標がかぶってたらオワオワリ
# l_xy = list(zip(x, y))
# for idx in range(len(l_xy)-1):
#     if l_xy[idx] == l_xy[idx+1]:
#         return self_inf

# 交わらないように
# for idx in range(len(l_xy)-1):
#     for jdx in range(len(l_xy)-2):
#         if judgeCross(l_xy[idx], l_xy[idx+1], l_xy[jdx], l_xy[jdx+2]):
#             return self_inf