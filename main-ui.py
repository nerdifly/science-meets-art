# lots of optimizations had to be made to get this to run at a decent speed

# the power of your pc may affect the performance of art
# main-UI.py

# this is realy slow, but it works and produces a gif
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
import matplotlib.cm as cm

# PARAMETERS
# When running interactively it's much faster to use fewer sample points.
FAST_MODE = True            # Set False to use full resolution (for exports)
INTERACTIVE_POINTS = 1200   # number of theta samples in FAST_MODE
THETA_FULL_POINTS = 4000
theta_full = np.linspace(0, 12*np.pi, THETA_FULL_POINTS)
if FAST_MODE:
    stride = max(1, int(len(theta_full) / INTERACTIVE_POINTS))
    theta = theta_full[::stride]
else:
    theta = theta_full

base_frames = 200
slow_factor = 5.5    # >1 = slower evolution
frames = int(base_frames * slow_factor)
color_speed = 0.1

# YOUR NAME
your_name = "Pablo Van der Sande"

# ORIGINAL FORMULA (text overlay)
formula_text = r"r = s * cos(g θ) * (2 + 0.5 * cos(21θ)) + sin(θ/3) * cos(7θ)"

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

# Always compute the global max using the full theta resolution to avoid
# clipping when switching between interactive and export modes.
global_max_r = compute_global_max_r(theta_full)

# CREATE LINE SEGMENTS FOR COLOR
def get_segments(x, y):
    # kept for compatibility but not used in the fast update path
    points = np.array([x, y]).T.reshape(-1,1,2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

# Precompute trig for the chosen theta to avoid repeated costly calls
cos_theta = np.cos(theta)
sin_theta = np.sin(theta)

# Preallocate a segments buffer to avoid allocating a new array each frame
segments_buffer = np.zeros((len(theta)-1, 2, 2), dtype=float)
color_base = theta[:-1]

# ANIMATION FUNCTION (interactive viewer)
def animate_loop_color_viewer():
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(-global_max_r*1.05, global_max_r*1.05)
    ax.set_ylim(-global_max_r*1.05, global_max_r*1.05)

    lc = LineCollection([], linewidths=2, cmap=cm.plasma)
    ax.add_collection(lc)

    # VSYNC / refresh control (best-effort). Many backends don't expose a
    # direct swap-interval control; we try a few common methods and safely
    # continue if none are available. Also set the animation interval to the
    # monitor refresh rate so frames are scheduled at ~60Hz.
    REFRESH_RATE = 60  # Hz, change if you target a different refresh rate
    interval_ms = int(round(1000.0 / REFRESH_RATE))

    def try_enable_vsync(fig):
        canvas = getattr(fig, 'canvas', None)
        if canvas is None:
            return False
        # Try canvas-level vsync setter
        for attr in ('set_vsync', 'setVSyncEnabled', 'set_vsync_enabled'):
            fn = getattr(canvas, attr, None)
            if callable(fn):
                try:
                    fn(True)
                    return True
                except Exception:
                    pass

        mgr = getattr(canvas, 'manager', None)
        if mgr is None:
            return False
        win = getattr(mgr, 'window', None)
        if win is None:
            return False

        # Try common window/context methods (Qt/OpenGL variants)
        candidates = ('set_swap_interval', 'setSwapInterval', 'set_vsync', 'setVSyncEnabled', 'swapInterval')
        for name in candidates:
            fn = getattr(win, name, None)
            if callable(fn):
                try:
                    # many APIs take 1 to enable vsync
                    fn(1)
                    return True
                except Exception:
                    try:
                        fn(True)
                        return True
                    except Exception:
                        pass

        # Try getting an OpenGL context and call setSwapInterval if available
        ctx_getters = ('context', 'getContext', 'opengl_context')
        for g in ctx_getters:
            getter = getattr(win, g, None)
            if callable(getter):
                try:
                    ctx = getter()
                    fn = getattr(ctx, 'setSwapInterval', None)
                    if callable(fn):
                        try:
                            fn(1)
                            return True
                        except Exception:
                            pass
                except Exception:
                    pass

        return False

    vsync_enabled = False
    try:
        vsync_enabled = try_enable_vsync(fig)
    except Exception:
        vsync_enabled = False
    # store a status message to show on-screen later (we'll set a text artist)
    vsync_status_msg = 'VSync: ON' if vsync_enabled else 'VSync: OFF (timed interval)'

    # Static text: formula and name
    ax.text(0, global_max_r*1.08, formula_text, color='white', fontsize=10, ha='center', va='bottom',
            fontweight='bold', backgroundcolor='black')
    ax.text(global_max_r*0.8, -global_max_r*1.05, your_name, color='white', fontsize=10, ha='right', va='bottom',
            fontweight='bold', backgroundcolor='black')

    # Dynamic text placeholders for s and g (marked animated for blitting)
    s_text = ax.text(-global_max_r*0.95, -global_max_r*1.05, "", color='white', fontsize=10, ha='left', va='bottom',
                     fontweight='bold', backgroundcolor='black', animated=True)
    g_text = ax.text(-global_max_r*0.95, -global_max_r*1.15, "", color='white', fontsize=10, ha='left', va='bottom',
                     fontweight='bold', backgroundcolor='black', animated=True)

    # FPS counter (exponential moving average) - low overhead
    fps_text = ax.text(global_max_r*0.95, global_max_r*1.05, "", color='white', fontsize=10, ha='right', va='top',
                       fontweight='bold', backgroundcolor='black', animated=True)
    # VSync status and live frame counter (on-screen feedback)
    status_text = ax.text(-global_max_r*0.95, global_max_r*1.05, vsync_status_msg, color='white', fontsize=10, ha='left', va='top',
                          fontweight='bold', backgroundcolor='black', animated=True)
    frame_text = ax.text(0, global_max_r*1.05, "", color='white', fontsize=10, ha='center', va='top',
                         fontweight='bold', backgroundcolor='black', animated=True)

    last_time = time.perf_counter()
    fps_ema = None
    fps_update_interval = 5  # update displayed FPS every N frames to reduce overhead

    def update(frame):
        # use nonlocal for the timing variables so we can update them
        nonlocal last_time, fps_ema

        s = s_func(frame)
        g = g_func(frame)

        # compute r using original formula but reuse precomputed trig
        r = original_r(theta, s, g)
        x = r * cos_theta
        y = r * sin_theta

        # fill segments_buffer in-place to avoid allocations
        segments_buffer[:,0,0] = x[:-1]
        segments_buffer[:,0,1] = y[:-1]
        segments_buffer[:,1,0] = x[1:]
        segments_buffer[:,1,1] = y[1:]

        lc.set_segments(segments_buffer)
        lc.set_array((color_base + frame*color_speed) % (2*np.pi))  # color animation

        # Update dynamic text
        s_text.set_text(f"s = {s:.2f}")
        g_text.set_text(f"g = {g:.3f}")

        # FPS measurement
        now = time.perf_counter()
        dt = now - last_time if last_time is not None else 0
        last_time = now
        if dt > 0:
            inst_fps = 1.0 / dt
            if fps_ema is None:
                fps_ema = inst_fps
            else:
                alpha = 0.1
                fps_ema = alpha * inst_fps + (1.0 - alpha) * fps_ema

        if frame % fps_update_interval == 0 and fps_ema is not None:
            fps_text.set_text(f"FPS: {fps_ema:.1f}")

        # Live frame counter updated every frame (cheap)
        frame_text.set_text(f"Frame: {frame+1}/{frames}")

        # status_text already set earlier to vsync_status_msg; leave it unchanged

        return lc, s_text, g_text, fps_text, status_text, frame_text

    # Keep a reference to the animation object to prevent it being garbage-collected
    anim = FuncAnimation(fig, update, frames=frames, blit=True, interval=interval_ms, repeat=True)
    # Ensure the event source uses our interval (some backends require setting it)
    try:
        anim.event_source.interval = interval_ms
    except Exception:
        pass

    # Show interactively (opens a viewer window when run in a desktop environment)
    plt.show()

    return anim


# RUN INTERACTIVE VIEWER
if __name__ == '__main__':
    animate_loop_color_viewer()