import math
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import minimize
from CubicSpline import Spline2D
import trapezoid_vel_profile as profile
import csv


# TODO
'''
・プロファイルパラメータも最適化したい
'''

'''
parameters
'''
# 走った
start = [0.0, 0.0, 0.0]  # x y Θ -> 0.54 0.35 0.0
point = [[0.0, 3.4, 0.0], [1.05, 3.5, 0.0], [1.1, 0.1, 0.0],
    [3.05, 0.5, 0.0], [3.0, 1.25, 0.0], 
    [2.2, 1.5, 0.0], [2.25, 2.25, 0.0],  
    [3.1, 2.5, 0.0], [2.75, 3.4, 0.0]]
estimate_number = [0, 1, 1, 1, 1, 1, 1, 1, 1]

# 遅いやつ
vel_acc_dec_vmax = [[-1.0, 2.5, 1.5, 2.0], [1.0, 1.0, 1.0, 1.0], [-1.0, 2.5, 1.5, 2.0], 
    [1.0, 2.0, 1.5, 1.5], [1.0, 1.5, 1.5, 1.0], 
    [1.0, 1.5, 1.5, 1.0], [1.0, 1.5, 1.5, 1.0], 
    [1.0, 1.5, 1.5, 1.0], [0.0, 1.0, 1.0, 1.0]]  # -1 --> stop
# 理論値 12.3s, 実測値 18s

# 速いやつ
# vel_acc_dec_vmax = [[-1.0, 3.5, 2.0, 2.5], [1.0, 1.0, 1.0, 2.0], [-1.0, 3.5, 2.0, 2.5], 
#     [1.5, 2.5, 2.0, 2.0], [1.5, 2.0, 1.5, 1.5], 
#     [1.5, 2.0, 1.5, 1.5], [1.5, 2.0, 1.5, 1.5], 
#     [1.5, 2.0, 1.5, 1.5], [0.0, 1.0, 1.0, 1.0]]  # -1 --> stop
# 理論値 11.8s, 実測値 16s

csv_name = "low"

# 戻る
# start = [3.1, 3.6, 0.0]  # x y Θ
# point = [[3.2, 0.0, 0.0], [1.25, 0.0, 0.0], [1.25, 3.25, 0.0],
#     [0.0, 3.25, 0.0], [0.0, 0.0, 0.0]]

# estimate_number = [0, 0, 0, 0, 0]

# vel_acc_dec_vmax = [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]  # -1 --> stop
# v_max = 1.0

# やばかった
# vel_acc_dec_vmax = [[-1.0, 3.0], [2.0, 2.0], [-1.0, 3.0], 
#     [2.0, 3.0], [2.0, 3.0], 
#     [2.0, 3.0], [2.0, 3.0], 
#     [2.0, 3.0], [0.0, 3.0]]  # -1 --> stop
# v_max = 3.0

zurasu = [0.01, 0.2]

frequency = 500  # hz

robot_size = 0.6 # m
roll_length = 0.3  # m

field = [[-0.55+zurasu[0], -0.55+zurasu[1]], [3.55+zurasu[0], 3.95+zurasu[1]]]  # lower_left to upper_right
wall_thick = 0.038
sz = [[-0.55+zurasu[0], -0.55+zurasu[1]], [0.8, 0.8]]  # point xy and size xy
cz = [[0.281+zurasu[0], -0.55+zurasu[1]], [0.45, 0.8]]
oz = [[-0.55+zurasu[0], 3.95+zurasu[1]], [0.5, -1.4]]
pz = [[2.31+zurasu[0], 3.95+zurasu[1]], [0.6, -0.5]]
bar1 = [[0.731+zurasu[0], -0.55+zurasu[1]], [wall_thick, 3.5]]
bar2 = [[1.631+zurasu[0], 3.95+zurasu[1]], [wall_thick, -3.5]]
poll1 = [[2.61-(0.118/2)+zurasu[0], 0.65-(0.118/2)+zurasu[1]], [0.118, 0.118]]
poll2 = [[2.61-(0.118/2)+zurasu[0], 1.65-(0.118/2)+zurasu[1]], [0.118, 0.118]]
poll3 = [[2.61-(0.118/2)+zurasu[0], 2.65-(0.118/2)+zurasu[1]], [0.118, 0.118]]

l_gain = 1
c_gain = 2

show_map = False

param = []
for i in range(len(point)):
    if i == 0:
        for j in range(estimate_number[i]):
            param.append(start[0]+(point[i][0]-start[0])
                         * (j+1)/(estimate_number[i]+1))
            param.append(start[1]+(point[i][1]-start[1])
                         * (j+1)/(estimate_number[i]+1))
    else:
        for j in range(estimate_number[i]):
            param.append(point[i-1][0]+(point[i][0]-point[i-1]
                         [0])*(j+1)/(estimate_number[i]+1))
            param.append(point[i-1][1]+(point[i][1]-point[i-1]
                         [1])*(j+1)/(estimate_number[i]+1))

# god adjustment
param[5] -= 0.2

fig, ax = plt.subplots(figsize=(6, 6))

'''
functions
'''


def spline(x, y):
    num = 100
    if len(x) < 2:
        print("you are stupid")
    if len(x) == 2:
        length = [np.hypot(x[0]-x[1], y[0]-y[1])]
        length_list = np.full(int(length[0]*num), length[0] / int(length[0]*num))
        l_x = np.linspace(x[0], x[1], int(length[0]*num+1))[:-1].tolist()
        l_y = np.linspace(y[0], y[1], int(length[0]*num+1))[:-1].tolist()
        l_k = np.zeros(int(length[0]*num)).tolist()
    else:
        sp = Spline2D(x, y)
        l_x, l_y, l_k, length = [], [], [], []
        length_list = np.array([])
        for i in range(len(x)-1):
            r_x, r_y = [], []
            x_tmp, y_tmp = [], []
            s = np.linspace(sp.s[i], sp.s[i+1], num+1)[:-1]
            for i_s in s:
                ix, iy = sp.calc_position(i_s)
                x_tmp.append(ix)
                y_tmp.append(iy)
                l_k.append(sp.calc_curvature(i_s))
            r_x.extend(x_tmp)
            r_y.extend(y_tmp)
            l_x.extend(x_tmp)
            l_y.extend(y_tmp)
            l = np.append(np.array([0]), np.hypot(np.diff(r_x), np.diff(r_y)))
            length_list = np.append(length_list, l)
            length.append(np.sum(l))

    return l_x, l_y, l_k, length, length_list


def calcTrajectory(param):
    nega = [0] + [i for i in range(len(vel_acc_dec_vmax))
                  if vel_acc_dec_vmax[i][0] < 0] + [len(vel_acc_dec_vmax)-1]
    r_x, r_y, r_k, length, length_list = [], [], [], [], []
    index_size_list = []
    p_x, p_y = [], []
    e = 0
    for i in range(len(nega)-1):
        px, py = [], []
        if i == 0:
            p_x.append(start[0])
            p_y.append(start[1])
            px.append(start[0])
            py.append(start[1])
        else:
            px.append(point[nega[i]][0])
            py.append(point[nega[i]][1])

        for j in range(nega[i]+int(i > 0), nega[i+1]+1):
            x_tmp, y_tmp = [], []
            if estimate_number[j] == 0:
                e -= 1
            for k in range(estimate_number[j]):
                e += k
                x_tmp.append(param[2*j + 2*e])
                y_tmp.append(param[2*j + 2*e + 1])
            p_x.extend(x_tmp + [point[j][0]])
            p_y.extend(y_tmp + [point[j][1]])
            px.extend(x_tmp + [point[j][0]])
            py.extend(y_tmp + [point[j][1]])

        x, y, k, l, list = spline(px, py)
        r_x.extend(x), r_y.extend(y), r_k.extend(
            k), length.extend(l), length_list.extend(list)
        index_size_list.append(len(x))

    if show_map:
        ax.cla()
        plt.gcf().canvas.mpl_connect('key_release_event',
                                     lambda event: [exit(0) if event.key == 'escape' else None])
        ax.grid(True)
        ax.axis("equal")

        # map settings
        field_map = []
        field_map.append(patches.Rectangle(xy=(field[0][0]-wall_thick, field[0][1]-wall_thick), width=field[1]
                         [0]-field[0][0]+wall_thick*2, height=field[1][1]-field[0][1]+wall_thick*2, angle=0, color="saddlebrown"))
        field_map.append(patches.Rectangle(xy=(field[0][0], field[0][1]), width=field[1]
                         [0]-field[0][0], height=field[1][1]-field[0][1], angle=0, color="limegreen"))

        field_map.append(patches.Rectangle(xy=(bar1[0][0]-(robot_size/2), bar1[0][1]), width=bar1[1][0]+(
            robot_size - 0.1), height=bar1[1][1]+(robot_size/2), angle=0, ec='navy', color="aliceblue"))
        field_map.append(patches.Rectangle(xy=(bar2[0][0]-(robot_size/2) - 0.1, bar2[0][1]), width=bar2[1][0]+(
            robot_size + 0.1), height=bar2[1][1]-(robot_size/2), angle=0, ec='navy', color="aliceblue"))

        field_map.append(patches.Rectangle(xy=(poll1[0][0]-(robot_size/2), poll1[0][1]-(robot_size/2)), width=poll1[1][0]+(
            robot_size), height=poll1[1][1]+(robot_size), angle=0, ec='navy', color="aliceblue"))
        field_map.append(patches.Rectangle(xy=(poll2[0][0]-(robot_size/2), poll2[0][1]-(robot_size/2)), width=poll2[1][0]+(
            robot_size), height=poll2[1][1]+(robot_size), angle=0, ec='navy', color="aliceblue"))
        field_map.append(patches.Rectangle(xy=(poll3[0][0]-(robot_size/2), poll3[0][1]-(robot_size/2)), width=poll3[1][0]+(
            robot_size), height=poll3[1][1]+(robot_size), angle=0, ec='navy', color="aliceblue"))

        field_map.append(patches.Rectangle(xy=(
            sz[0][0], sz[0][1]), width=sz[1][0], height=sz[1][1], angle=0, ec='navy', color="red"))
        field_map.append(patches.Rectangle(xy=(
            cz[0][0], cz[0][1]), width=cz[1][0], height=cz[1][1], angle=0, ec='navy', color="blue"))
        field_map.append(patches.Rectangle(xy=(
            oz[0][0], oz[0][1]), width=oz[1][0], height=oz[1][1], angle=0, ec='navy', color="darkturquoise"))
        field_map.append(patches.Rectangle(xy=(
            pz[0][0], pz[0][1]), width=pz[1][0], height=pz[1][1], angle=0, ec='navy', color="gold"))

        field_map.append(patches.Rectangle(xy=(
            bar1[0][0], bar1[0][1]), width=bar1[1][0], height=bar1[1][1], angle=0, ec='navy', color="saddlebrown"))
        field_map.append(patches.Rectangle(xy=(
            bar2[0][0], bar2[0][1]), width=bar2[1][0], height=bar2[1][1], angle=0, ec='navy', color="saddlebrown"))

        field_map.append(patches.Rectangle(xy=(
            poll1[0][0], poll1[0][1]), width=poll1[1][0], height=poll1[1][1], angle=0, ec='navy', color="burlywood"))
        field_map.append(patches.Rectangle(xy=(
            poll2[0][0], poll2[0][1]), width=poll2[1][0], height=poll2[1][1], angle=0, ec='navy', color="burlywood"))
        field_map.append(patches.Rectangle(xy=(
            poll3[0][0], poll3[0][1]), width=poll3[1][0], height=poll3[1][1], angle=0, ec='navy', color="burlywood"))

        field_map.append(patches.Rectangle(xy=(-robot_size/2, -robot_size/2),
                         width=robot_size, height=robot_size, angle=0, ec='gold', fill=False))

        [ax.add_patch(m) for m in field_map]

        ax.scatter(r_x, r_y, s=10, label="traj", c="dimgray")
        ax.scatter(p_x, p_y, s=10, label="traj", c="orange")
        plt.pause(0.001)

    return p_x, p_y, r_x, r_y, r_k, length, length_list, index_size_list


def judgeCross(a, b, c, d):
    if (((a[0] - b[0]) * (c[1] - a[1]) + (a[1] - b[1]) * (a[0] - c[0])) * ((a[0] - b[0]) * (d[1] - a[1]) + (a[1] - b[1]) * (a[0] - d[0])) < 0):
        if (((c[0] - d[0]) * (a[1] - c[1]) + (c[1] - d[1]) * (c[0] - a[0])) * ((c[0] - d[0]) * (b[1] - c[1]) + (c[1] - d[1]) * (c[0] - b[0])) < 0):
            return True

# 目的関数


def func(param):
    r_k, length = calcTrajectory(param)[4:6]
    cost = np.hypot(np.sum(length)*l_gain, np.max(np.abs(r_k))*c_gain)
    print(cost)

    return cost


'''
制約条件式
'''
# 外枠干渉回避


def over_cons(param):
    r_x, r_y = calcTrajectory(param)[2:4]
    over = [np.min(r_x) - (field[0][0]+robot_size*0.5), np.min(r_y) - (field[0][1]+robot_size*0.5),
            field[1][0]-robot_size*0.5 - (np.max(r_x)), field[1][1]-robot_size*0.5 - (np.max(r_y))]
    if (np.any(np.array(over) < 0)):
        print("over_error : {}".format(over))
    return over

# obstacle回避


def obstacle_cons(param):
    r_x, r_y = calcTrajectory(param)[2:4]
    num = 0
    for x, y in zip(r_x, r_y):
        if (bar1[0][0]-(robot_size/2)) < x < ((bar1[0][0]+bar1[1][0])+(robot_size/2) - 0.1) and y < (bar1[0][1]+bar1[1][1]+(robot_size/2)) \
                or (bar2[0][0]-(robot_size/2)) < x < ((bar2[0][0]+bar2[1][0])+(robot_size/2) + 0.1) and (bar2[0][1]+bar2[1][1]-(robot_size/2)) < y \
                or (poll1[0][0]-(robot_size/2)) < x < ((poll1[0][0]+poll1[1][0])+(robot_size/2)) and (poll1[0][1]-(robot_size/2)) < y < ((poll1[0][1]+poll1[1][1])+(robot_size/2)) \
                or (poll2[0][0]-(robot_size/2)) < x < ((poll2[0][0]+poll2[1][0])+(robot_size/2)) and (poll2[0][1]-(robot_size/2)) < y < ((poll2[0][1]+poll2[1][1])+(robot_size/2)) \
                or (poll3[0][0]-(robot_size/2)) < x < ((poll3[0][0]+poll3[1][0])+(robot_size/2)) and (poll3[0][1]-(robot_size/2)) < y < ((poll3[0][1]+poll3[1][1])+(robot_size/2)):
            num -= 1
    if num < 0:
        print("obstacle_error")
    return num

# 交差回避


def cross_cons(param):
    p_x, p_y = calcTrajectory(param)[:2]

    p = [[x, y] for x, y in zip(p_x, p_y)]
    num = 0
    for idx in range(len(p)-1):
        for jdx in range(len(p)-2):
            if judgeCross(p[idx], p[idx+1], p[jdx+1], p[jdx+2]):
                num -= 1
    if num < 0:
        print("cross_error")
    return num

# 右行け左行け
# TODO


def checkKeta(c):
    if c == '0.0':
        return '0.000'
    else:
        return c

def main():
    # 制約条件式が0以上になるようにする
    # ineq 不等号, eq   等号
    cons = (
        {'type': 'ineq', 'fun': over_cons},
        {'type': 'ineq', 'fun': obstacle_cons},
        {'type': 'ineq', 'fun': cross_cons}
    )

    result = minimize(func, x0=param, constraints=cons, method="COBYLA")
    print(result)

    p_x, p_y, r_x, r_y, _, length, length_list, index_size_list = calcTrajectory(result.x)

    '''
    plot parameters
    '''
    # fig = plt.figure(figsize=(10, 8))
    ax1 = plt.subplot2grid((2, 3), (0, 0))
    ax2 = plt.subplot2grid((2, 3), (0, 1))
    ax3 = plt.subplot2grid((2, 3), (0, 2))
    ax4 = plt.subplot2grid((2, 3), (1, 0))
    ax5 = plt.subplot2grid((2, 3), (1, 1))
    ax6 = plt.subplot2grid((2, 3), (1, 2))

    '''
    calc trapezoid velocity
    '''
    x_start, x_target = 0.0, 0.0
    v_start, v_target = 0.0, 0.0
    t_list = []
    start_num = [0] + [i for i in range(len(vel_acc_dec_vmax))
                  if vel_acc_dec_vmax[i][0] < 0]
    e = 0

    for i in range(len(vel_acc_dec_vmax)):
        x_target += length[i*2+e]
        for j in range(estimate_number[i]):
            x_target += length[i*2+e+j+1]

        vel_acc_dec_vmax[i][0] = 0 if vel_acc_dec_vmax[i][0] < 0 else vel_acc_dec_vmax[i][0]
        v_target = vel_acc_dec_vmax[i][0]
        (Y, Yd, _, t) = profile.plan(x_start, x_target, v_start,
                                     v_target, vel_acc_dec_vmax[i][3], vel_acc_dec_vmax[i][1], vel_acc_dec_vmax[i][2], frequency)
        x_start += length[i*2+e]
        for j in range(estimate_number[i]):
            x_start += length[i*2+e+j+1]

        v_start = vel_acc_dec_vmax[i][0]
        t_list.append(t[-1])
        e += estimate_number[i]-1

    '''
    calc omega
    '''
    theta = [start[2]] + [point[i][2] for i in range(len(point))]
    omega = []
    for i in range(len(theta)-1):
        omega.append(math.radians(2*(abs(theta[i+1] - theta[i])) / t_list[i]) * roll_length
                     if theta[i] != theta[i+1] else 0.0)

    print('Max Omega is {}'.format(omega))

    '''
    adjustment velocity
    '''
    plot_t = []
    x_start, x_target = 0.0, 0.0
    v_start, v_target = 0.0, 0.0
    t_list = []
    e = 0
    p = 0
    # for save data
    stack_t = 0
    stack_y = []
    stack_len = []

    for i in range(len(vel_acc_dec_vmax)):
        x_target += length[i*2+e]
        for j in range(estimate_number[i]):
            x_target += length[i*2+e+j+1]
        v_target = vel_acc_dec_vmax[i][0] - omega[i] * vel_acc_dec_vmax[i][0] / vel_acc_dec_vmax[i][3]
        (Y, Yd, _, t) = profile.plan(x_start, x_target, v_start,
                                    v_target, vel_acc_dec_vmax[i][3]-omega[i], vel_acc_dec_vmax[i][1], vel_acc_dec_vmax[i][2], frequency)
        x_start += length[i*2+e]
        x_start += sum([
            length[i*2+e+j+1] for j in range(estimate_number[i])
        ])
        v_start = vel_acc_dec_vmax[i][0] - omega[i] * vel_acc_dec_vmax[i][0] / vel_acc_dec_vmax[i][3]
        t_list.append(t[-1])
        
        ax1.plot(stack_t+t, Y, label=str(i))  # Pos
        ax2.plot(stack_t+t, Yd, label=str(i))  # Vel
        
        if np.shape(stack_y)[0] != p+1:
            stack_y.append(Y[:])
            stack_len.append(length[i*2+e])
            stack_len[p] += sum([
                length[i*2+e+j+1] for j in range(estimate_number[i])
            ])
            stack_t = 0
            plot_t.append([0.0])
            plot_t[p].extend(stack_t+t[1:])
        else:
            stack_y[p].extend(Y[1:])
            stack_len[p] += length[i*2+e]
            stack_len[p] += sum([
                length[i*2+e+j+1] for j in range(estimate_number[i])
            ])
            plot_t[p].extend(stack_t+t[1:])
        if len(start_num) > p+1:
            if start_num[p+1] == i: p+=1

        e += estimate_number[i]-1
        stack_t += t[-1]

    stack_y = np.array(stack_y)

    '''
    adjustment angular velocity
    '''
    theta = [start[2]] + [point[i][2] for i in range(len(point))]
    omega = []
    th_start, th_target = 0.0, 0.0
    p = 0
    stack_t = 0
    stack_th = []

    for i in range(len(theta)-1):
        if theta[i] != theta[i+1]:
            th_target += abs(theta[i+1] - theta[i])
            omega_max = 2*(abs(theta[i+1] - theta[i])) / t_list[i]
            omega.append(math.radians(omega_max) * roll_length)
            angular_acc = 2*omega_max / t_list[i]

            (TH, THd, _, t) = profile.plan(np.deg2rad(th_start), np.deg2rad(th_target), 0, 0,
                                           np.deg2rad(omega_max), np.deg2rad(angular_acc), np.deg2rad(angular_acc), frequency)

            ax4.plot(stack_t+t, TH, label=str(i))  # Pos
            ax5.plot(stack_t+t, THd, label=str(i))  # Vel
            th_start += abs(theta[i+1] - theta[i])
            stack_t += t[-1]
            if np.shape(stack_th)[0] != p+1:
                stack_th.append(TH[:])
            else:
                stack_th[p].extend(TH[1:])
            if len(start_num) > p+1:
                if start_num[p+1] == i: p+=1
        else:
            omega.append(0.0)
            if np.shape(stack_th)[0] != p+1:
                stack_th.append([0.0] * (int(t_list[i]*frequency)+1))
            else:
                stack_th[p].extend([0.0] * int(t_list[i]*frequency))
            if len(start_num) > p+1:
                if start_num[p+1] == i: p+=1
    
    stack_th = np.array(stack_th)

    '''
    fusion
    '''
    trajectory = []
    length_cum = np.cumsum(length_list)
    index_size_cum = np.cumsum(index_size_list)

    last_index = 0
    for j in range(len(stack_y)):
        out_trajectory = []
        count = 1
        pre_index = 0
        for y, th in zip(stack_y[j], stack_th[j]):
            idx = np.abs(np.asarray(length_cum) - y).argmin()
            if (idx != last_index and pre_index != idx) or idx == index_size_cum[j]-1:
                if count > 1:
                    dx = (r_x[idx]-r_x[pre_index])/count
                    dy = (r_y[idx]-r_y[pre_index])/count
                    for i in range(count):
                        trajectory.append((r_x[pre_index]+dx*i, r_y[pre_index]+dy*i, th))
                        out_trajectory.append((r_x[pre_index]+dx*i, r_y[pre_index]+dy*i, th))
                    count = 1
                    if idx == index_size_cum[j]-1:
                        trajectory.append((r_x[idx], r_y[idx], th))
                        out_trajectory.append((r_x[idx], r_y[idx], th))
                else:
                    trajectory.append((r_x[idx], r_y[idx], th))
                    out_trajectory.append((r_x[idx], r_y[idx], th))
            else:
                count += 1
            pre_index = idx
        last_index = pre_index

        print(len(stack_y[j]))
        print(len(plot_t[j]))
        print(len(out_trajectory))

        t = checkKeta(str(round(plot_t[j][0], 3)))
        x = checkKeta(str(round(out_trajectory[0][0]*1000, 3)))
        y = checkKeta(str(round(out_trajectory[0][1]*1000, 3)))
        th = checkKeta(str(round(out_trajectory[0][2]*1000, 3)))
        vx = checkKeta(str(round(0.0, 3)))
        vy = checkKeta(str(round(0.0, 3)))
        v = checkKeta(str(round(0.001, 3)))
        argv = str(math.atan2(out_trajectory[1][1]-out_trajectory[0][1], out_trajectory[1][0]-out_trajectory[0][0]))
        omega = checkKeta(str(round(0.0, 3)))

        push_data = [[t, x, y, th, vx, vy, v, argv, omega]]
        for i in range(1, len(plot_t[j])):
            t = checkKeta(str(round(plot_t[j][i], 3)))
            x = checkKeta(str(round(out_trajectory[i][0]*1000, 3)))
            y = checkKeta(str(round(out_trajectory[i][1]*1000, 3)))
            th = checkKeta(str(round(out_trajectory[i][2]*1000, 3)))
            vx = checkKeta(str(round((out_trajectory[i][0]-out_trajectory[i-1][0])*frequency*1000, 3)))
            vy = checkKeta(str(round((out_trajectory[i][1]-out_trajectory[i-1][1])*frequency*1000, 3)))
            v = checkKeta(str(round(np.hypot(out_trajectory[i][0]-out_trajectory[i-1][0], out_trajectory[i][1]-out_trajectory[i-1][1])*frequency*1000, 3)))
            argv = checkKeta(str(math.atan2(out_trajectory[i][1]-out_trajectory[i-1][1], out_trajectory[i][0]-out_trajectory[i-1][0])))
            omega = checkKeta(str(round((out_trajectory[i][2]-out_trajectory[i-1][2])*frequency*1000, 3)))
            push_data.append([t, x, y, th, str(vx), str(vy), str(v), str(argv), str(omega)])

        with open(csv_name+str(j)+'.csv', 'w', newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['LABEL t', 'x', 'y', 'theta', 'vx', 'vy', '|v|', 'arg(v)', 'omega'])
            writer.writerow(['INTERVALTIME 0.001'])
            writer.writerow(['START 0.000', 
                            checkKeta(str(round(out_trajectory[0][0]*1000, 3))), 
                            checkKeta(str(round(out_trajectory[0][1]*1000, 3))), 
                            checkKeta(str(round(out_trajectory[0][2]*1000, 3))), 
                            '0.000', '0.000', '0.001', checkKeta(str(math.atan2(out_trajectory[3][1]-out_trajectory[0][1], out_trajectory[3][0]-out_trajectory[0][0]))), '0.000'])
            writer.writerow(['END '+checkKeta(str(round(plot_t[j][-1], 3))), 
                            checkKeta(str(round(out_trajectory[-1][0]*1000, 3))), 
                            checkKeta(str(round(out_trajectory[-1][1]*1000, 3))), 
                            '0.000', '0.000', '0.000', '0.001', '0.000', '0.000'])
            writer.writerow(['LENGTH '+str(stack_len[j]*1000)])
            writer.writerow(['TIME '+str(plot_t[j][-1])])
            writer.writerow([])
            writer.writerow(['MAIN'])
            writer.writerows(push_data)

    ax6.plot(trajectory)

    '''
    plot
    '''
    print('m : {}'.format(sum(length)))
    print('s : {}'.format(sum(t_list)))

    ax3.scatter([list(tup) for tup in zip(*trajectory)][0], [list(tup)
                for tup in zip(*trajectory)][1], s=5, label="trajectory")
    ax3.scatter(p_x, p_y, s=30, label="point")
    ax3.axis('square')
    ax3.grid(which='major', linestyle='-')
    # if show_map:
    plt.show()


if __name__ == '__main__':
    main()