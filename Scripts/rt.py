import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import NN

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def animate(i):
    ax1.clear()
    ax1.plot(NN.x,NN.y)
ani = animation.FuncAnimation(fig, animate, interval=10)
plt.show()