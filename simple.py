import numpy as np
import matplotlib.pyplot as plt

from task.noholo import *

drawf(np.eye(3))
drawrobot(np.eye(3), 0.1)
drawrobot(mktr(0.5, 0.3) @ mkrot(np.pi/4), 0.1)
plt.gca().axis("equal")
plt.show()