import matplotlib.pyplot as plt
import pandas as pd
import glob
import sys
from pathlib import Path
import matplotlib.animation as animation


fig, ax = plt.subplots(figsize = (5, 5))
ax.set_xlabel("X", fontsize=13)
ax.set_ylabel("Y", fontsize=13)

min_x, max_x = float('inf'), 0
min_y, max_y = float('inf'), 0
x, y = [], []
for csv in glob.glob(str(Path(sys.argv[0])) + '/../*.csv'):
    df = pd.read_csv(csv, engine='python', skipinitialspace=True)
    if min_x > min(df['x'][6:]): min_x = min(df['x'][6:])
    if max_x < max(df['x'][6:]): max_x = max(df['x'][6:])
    if min_y > min(df['y'][6:]): min_y = min(df['y'][6:])
    if max_y < max(df['y'][6:]): max_y = max(df['y'][6:])
    ax.set_xlim(min_x-100, max_x+100)
    ax.set_ylim(min_y-100, max_y+100)
    x.extend(df['x'][6:])
    y.extend(df['y'][6:])

line, = ax.plot([], [], lw=2)
xdata, ydata = [], []

def run(i):
    xdata.append(x[i])
    ydata.append(y[i])
    line.set_data(xdata, ydata)

ani = animation.FuncAnimation(fig, run, interval=10, repeat=False, frames=int(len(x)-1))
# w = animation.PillowWriter(fps=10)
# ani.save('test.gif', writer=w)
plt.title("RouteGUI plot", fontsize=15)
plt.show()
