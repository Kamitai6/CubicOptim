'''
# Copyright (c) 2018 Paul Guénette
# Copyright (c) 2018 Oskar Weigl

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# This algorithm is based on:
# FIR filter-based online jerk-constrained trajectory generation
# https://www.researchgate.net/profile/Richard_Bearee/publication/304358769_FIR_filter-based_online_jerk-controlled_trajectory_generation/links/5770ccdd08ae10de639c0ff7/FIR-filter-based-online-jerk-controlled-trajectory-generation.pdf
'''
import numpy as np
import math
import matplotlib.pyplot as plt


def calc(Xi, Xf, Vi, Vf, Vmax, Amax, Dmax):
    dX = Xf - Xi                   # Distance to travel
    stop_dist = Vi**2 / (2*Dmax)   # Minimum stopping distance
    dXstop = np.sign(Vi)*stop_dist # Minimum stopping displacement
    s = np.sign(dX - dXstop)       # Sign of coast velocity (if any)
    Ar =  s*Amax                   # Maximum Acceleration (signed)
    Dr = -s*Dmax                   # Maximum Deceleration (signed)
    Vr =  s*Vmax                   # Maximum Velocity (signed)

    if s*Vi > s*Vr:
        print("Handbrake!")
        Ar = -s*Amax

    Ta =  (Vr-Vi)/Ar
    Td = -(Vr-Vf)/Dr

    dXmin = Ta*(Vr+Vi)/2.0 + Td*(Vr+Vf)/2.0

    if s*dX < s*dXmin:
        # print("Short Move:")
        Vr = s*math.sqrt((-Ar*Vf**2 + Dr*Vi**2 + 2*Ar*Dr*dX) / (Dr-Ar))
        Ta = max(0,  (Vr-Vi)/Ar)
        Td = max(0, -(Vr-Vf)/Dr)
        Tv = 0
    else:
        # print("Long move:")
        Tv = (dX - dXmin)/Vr

    Tf = Ta+Tv+Td

    return (Ar, Vr, Dr, Ta, Tv, Td, Tf)

def plan(Xi, Xf, Vi, Vf, Vmax, Amax, Dmax, Freq):
    Ar, Vr, Dr, Ta, Tv, Td, Tf = calc(Xi, Xf, Vi, Vf, Vmax, Amax, Dmax)

    t_traj = np.arange(0, Tf+1/Freq, 1/Freq)
    y   = [None]*len(t_traj)
    yd  = [None]*len(t_traj)
    ydd = [None]*len(t_traj)

    y_Accel = Xi + Vi*Ta + 0.5*Ar*Ta**2

    for i in range(len(t_traj)):
        t = t_traj[i]
        if t < 0: # Initial conditions
            y[i]   = Xi
            yd[i]  = Vi
            ydd[i] = 0
        elif t < Ta: # Acceleration
            y[i]   = Xi + Vi*t + 0.5*Ar*t**2
            yd[i]  = Vi + Ar*t
            ydd[i] = Ar
        elif t < Ta+Tv: # Coasting
            y[i]   = y_Accel + Vr*(t-Ta)
            yd[i]  = Vr
            ydd[i] = 0
        elif t < Tf: # Deceleration
            td     = t-Tf
            y[i]   = Xf + Vf*td + 0.5*Dr*td**2
            yd[i]  = Vf + Dr*td
            ydd[i] = Dr
        elif t >= Tf: # Final condition
            y[i]   = Xf
            yd[i]  = Vf
            ydd[i] = 0
        else:
            raise ValueError("t = {} is outside of considered range".format(t))
        
    # print("Final time is {}".format(Tf))

    return (y, yd, ydd, t_traj)

def graphical_test():
    Xi = 0
    Xf = 5.5
    Vi = 0
    Vf = 5
    Vmax = 10
    Amax = 10
    Dmax = 50
    (Y, Yd, Ydd, t) = plan(Xi, Xf, Vi, Vf, Vmax, Amax, Dmax)

    plt.plot([t[0], t[-1]], [Vmax, Vmax], 'g--')
    plt.plot([t[0], t[-1]], [-Vmax, -Vmax], 'g--')

    plt.plot(t, Y) # Pos
    plt.plot(t, Yd) # Vel
    plt.plot(0, Xi, 'bo') # Pos Initial
    plt.plot(0, Vi, 'ro') # Vel Initial

    plt.grid()
    plt.show()

if __name__ == '__main__':
    graphical_test()