from cmath import inf
import math
from re import A
from this import d
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from CubicSpline import Spline2D#, QuinticPolynomial

start = [0.0, 0.0]
mid =  [[-0.1, 3.0], [0.8, 3.5], [0.8, 0.0], [2.8, 0.6], [1.8, 1.4], [2.8, 2.6]]
goal =  [2.8, 4.0]

field = [10, 10]

curva_gain = 1e-2

def spline(x, y):
    num=1000
    sp = Spline2D(x, y)
    s = np.linspace(0, sp.s[-1], num+1)[:-1]
    r_x, r_y, r_vx, r_vy, r_ax, r_ay, r_k = [], [], [], [], [], [], []
    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        r_x.append(ix)
        r_y.append(iy)
        r_k.append(sp.calc_curvature(i_s))
        dx, dy = sp.calc_velocity(i_s)
        r_vx.append(dx)
        r_vy.append(dy)
        ddx, ddy = sp.calc_acceleration(i_s)
        r_ax.append(ddx)
        r_ay.append(ddy)
    travel = np.cumsum([np.hypot(dx, dy) for dx, dy in zip(np.diff(r_x), np.diff(r_y))]).tolist()
    travel = np.concatenate([[0.0], travel])

    # p = QuinticPolynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)
    return r_x, r_y, r_vx, r_vy, r_ax, r_ay, r_k, travel

def judgeCross(a, b, c, d):
    if (((a[0] - b[0]) * (c[1] - a[1]) + (a[1] - b[1]) * (a[0] - c[0])) * ((a[0] - b[0]) * (d[1] - a[1]) + (a[1] - b[1]) * (a[0] - d[0])) < 0):
        if (((c[0] - d[0]) * (a[1] - c[1]) + (c[1] - d[1]) * (c[0] - a[0])) * ((c[0] - d[0]) * (b[1] - c[1]) + (c[1] - d[1]) * (c[0] - b[0])) < 0):
            return True

# 目的関数
def func(param):
    param_xy = param
    x = [start[0]]
    y = [start[1]]
    x.append(param_xy[0])
    y.append(param_xy[1])

    for i in range(len(mid)):
        x.append(mid[i][0])
        y.append(mid[i][1])
        x.append(param_xy[2*i+2])
        y.append(param_xy[2*i+3])

    x.append(goal[0])
    y.append(goal[1])

    print("{}, {}".format(x, y))

    # 座標がかぶってたらオワオワリ
    l_xy = list(zip(x, y))
    for idx in range(len(l_xy)-1):
        if l_xy[idx] == l_xy[idx+1]:
            return 1e+7
    
    # 交わらないように
    for idx in range(len(l_xy)-1):
        for jdx in range(len(l_xy)-2):
            if judgeCross(l_xy[idx], l_xy[idx+1], l_xy[jdx], l_xy[jdx+2]):
                return 1e+7

    r_x, r_y, r_vx, r_vy, r_ax, r_ay, r_k, travel = spline(x, y)

    if max(np.abs(r_x)) >  field[0] or max(np.abs(r_y)) >  field[1]:
        return 1e+7
    if min(np.abs(r_x)) < -field[0] or min(np.abs(r_y)) < -field[1]:
        return 1e+7

    return travel[-1]*1 + max(np.abs(r_k))*curva_gain

# 制約条件式
def min_cons(param):
    return param + max(field[0], field[1])

def max_cons(param):
    return max(field[0], field[1]) - param

# 制約条件式が0以上になるようにする
# ineq 不等号
# eq   等号
cons = (
    {'type': 'ineq', 'fun': min_cons},
    {'type': 'ineq', 'fun': max_cons}
)

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

result = minimize(func, x0=param, constraints=cons, method="COBYLA")

print(result.x)

result_xy = result.x
x = [start[0]]
y = [start[1]]
x.append(result_xy[0])
y.append(result_xy[1])

for i in range(len(mid)):
    x.append(mid[i][0])
    y.append(mid[i][1])
    x.append(result_xy[2*i+2])
    y.append(result_xy[2*i+3])

x.append(goal[0])
y.append(goal[1])

r_x, r_y, r_vx, r_vy, r_ax, r_ay, r_k, travel = spline(x, y)
plt.plot(r_x, r_y, label="interp1d")
plt.grid(which='major', linestyle='-')
plt.show()