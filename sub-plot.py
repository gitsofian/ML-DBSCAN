import numpy as np
from matplotlib import pyplot as plt

x = np.linspace(0, 2*np.pi, 400)
y = np.sin(x**2)

fig, axs = plt.subplots( 2,2)
fig.set_facecolor(color="black")

axs[0,0].set_facecolor("w")
axs[0,0].plot(x,y)
axs[0,0].grid()



axs[0,1].set_facecolor("b")
axs[1,0].set_facecolor("orange")
axs[1,1].set_facecolor("g")

plt.show()
