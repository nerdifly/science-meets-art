# main.py
# the too fast moving animation but i will leave it here for reference
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import matplotlib.cm as cm

# PARAMETERS
theta = np.linspace(0, 12*np.pi, 4000)  # 0 to 12π
frames = 200  # number of frames for a loop

# ORIGINAL FORMULA
def original_r(theta, s, g):
    return s * np.cos(g * theta) * (2 + 0.5 * np.cos(21*theta)) + np.sin(theta/3) * np.cos(7*theta)

# LOOPABLE s and g (sinusoidal)
def s_func(t):
    return 10 * np.sin(2 * np.pi * t / frames)  # -10 to 10

def g_func(t):
    return 0.25 + 0.25 * np.sin(2 * np.pi * t / frames)  # 0 to 0.5

# CALCULATE MAX RADIUS
def compute_global_max_r(theta):
    max_r = 0
    t_samples = np.linspace(0, frames, 20)
    for t in t_samples:
        s = s_func(t)
        g = g_func(t)
        r = original_r(theta, s, g)
        max_r = max(max_r, np.max(np.abs(r)))
    return max_r

global_max_r = compute_global_max_r(theta)

# CREATE LINE SEGMENTS FOR COLOR GRADIENT
def get_segments(x, y):
    points = np.array([x, y]).T.reshape(-1,1,2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

# ANIMATION FUNCTION
def animate_loop_color(filename):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(-global_max_r*1.05, global_max_r*1.05)
    ax.set_ylim(-global_max_r*1.05, global_max_r*1.05)

    # Use a LineCollection for gradient colors
    lc = LineCollection([], linewidths=2, cmap=cm.plasma)
    ax.add_collection(lc)

    def update(frame):
        s = s_func(frame)
        g = g_func(frame)
        r = original_r(theta, s, g)
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        segments = get_segments(x, y)
        lc.set_segments(segments)
        # Color gradient along theta + small time shift for animation
        lc.set_array((theta + frame*0.1) % (2*np.pi))
        return lc,

    anim = FuncAnimation(fig, update, frames=frames, blit=True)
    anim.save(filename, fps=30, dpi=200)
    plt.close(fig)

# RENDER LOOPABLE COLOR ANIMATION
animate_loop_color("original_animation_loopable_color.mp4")