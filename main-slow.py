# main-slow.py
# this is the final version for an mp4
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import matplotlib.cm as cm

# PARAMETERS
theta = np.linspace(0, 12*np.pi, 4000)
base_frames = 200
slow_factor = 3
frames = int(base_frames * slow_factor)
color_speed = 0.1

# YOUR NAME
your_name = "Pablo Van der Sande"

# ORIGINAL FORMULA (as text for overlay)
formula_text = r"r = s · cos(g θ) · (2 + 0.5 · cos(21θ)) + sin(θ/3) · cos(7θ)"

# ORIGINAL FORMULA FUNCTION
def original_r(theta, s, g):
    return s * np.cos(g * theta) * (2 + 0.5 * np.cos(21*theta)) + np.sin(theta/3) * np.cos(7*theta)

# LOOPABLE s and g
def s_func(t):
    return 10 * np.sin(2 * np.pi * t / frames)

def g_func(t):
    return 0.25 + 0.25 * np.sin(2 * np.pi * t / frames)

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

# CREATE LINE SEGMENTS FOR COLOR
def get_segments(x, y):
    points = np.array([x, y]).T.reshape(-1,1,2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

# ANIMATION FUNCTION
def animate_loop_color_with_text(filename):
    fig, ax = plt.subplots(figsize=(9,9))
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(-global_max_r*1.05, global_max_r*1.05)
    ax.set_ylim(-global_max_r*1.05, global_max_r*1.05)

    lc = LineCollection([], linewidths=2, cmap=cm.plasma)
    ax.add_collection(lc)

    # Static text: formula (top center)
    ax.text(0, global_max_r*1.08, formula_text,
            color='white', fontsize=18, ha='center', va='bottom',
            fontweight='bold', backgroundcolor='black', alpha=0.6)

    # Name (bottom right)
    ax.text(global_max_r*0.95, -global_max_r*1.05, your_name,
            color='white', fontsize=14, ha='right', va='bottom',
            fontweight='bold', backgroundcolor='black', alpha=0.6)

    # Dynamic text for s and g (bottom left)
    s_text = ax.text(-global_max_r*0.95, -global_max_r*1.05, "", 
                     color='white', fontsize=14, ha='left', va='bottom',
                     fontweight='bold', backgroundcolor='black', alpha=0.6)
    g_text = ax.text(-global_max_r*0.95, -global_max_r*1.15, "", 
                     color='white', fontsize=14, ha='left', va='bottom',
                     fontweight='bold', backgroundcolor='black', alpha=0.6)

    def update(frame):
        s = s_func(frame)
        g = g_func(frame)
        r = original_r(theta, s, g)
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        segments = get_segments(x, y)
        lc.set_segments(segments)
        lc.set_array((theta + frame*color_speed) % (2*np.pi))

        # Update dynamic text
        s_text.set_text(f"s = {s:.2f}")
        g_text.set_text(f"g = {g:.3f}")
        return lc, s_text, g_text

    anim = FuncAnimation(fig, update, frames=frames, blit=True)
    anim.save(filename, fps=30, dpi=200)
    plt.close(fig)

# Render animation
animate_loop_color_with_text("original_animation_polished.mp4")