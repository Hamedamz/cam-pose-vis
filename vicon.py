import json

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import RangeSlider, Button
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
from matplotlib import cm
import matplotlib

matplotlib.use('TkAgg')

# with tune vel:
# logs/vicon_09_40_19_08_06_2025.json
# logs/vicon_09_46_28_08_06_2025.json 10% higher PID
# with minimized pid
# logs/vicon_09_42_48_08_06_2025.json

show_markers = False
aspect_equal = False
load_csv = False

if load_csv:
    # --- Load the CSV ---
    file_path = '/Users/hamed/Desktop/N_240_wba_3s/FLS_LW_N_240_WBA3S.csv'
    df = pd.read_csv(file_path, skiprows=4).dropna(how='all')
    df.columns = ['Frame', 'Sub Frame', 'RX', 'RY', 'RZ', 'TX', 'TY', 'TZ']
    df = df.apply(pd.to_numeric, errors='coerce').dropna()

    x, y, z = df['TX'].values/1000, df['TY'].values/1000, df['TZ'].values/1000

else:
    dir = "vicon/"
    # dir = "/Users/hamed/Desktop/logs_2/"
    with open(dir + "vicon_09_40_19_08_06_2025.json", 'r') as f:  # cf
        frames = json.load(f)["frames"]

    scale = 1

    print(len(frames), frames[0], frames[-1])
    x = [frame["tvec"][0] * scale for frame in frames]
    y = [frame["tvec"][1] * scale for frame in frames]
    z = [frame["tvec"][2] * scale for frame in frames]


# --- Create gradient segments ---
def create_colored_line_segments(x, y, z, cmap=cm.viridis):
    points = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    colors = cmap(np.linspace(0, 1, len(segments)))
    return segments, colors


# Initial segments and colors
segments, colors = create_colored_line_segments(x, y, z)

# --- Set up plot ---
fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(121, projection='3d')
plt.subplots_adjust(bottom=0.3)

ax_stats = fig.add_subplot(122)
ax_stats.axis('off')  # Hide axes
stats_text = ax_stats.text(0, 1, '', verticalalignment='top', fontsize=10, family='monospace')

lc = Line3DCollection(segments, colors=colors, linewidth=2)
ax.add_collection3d(lc)

if show_markers:
    start_marker = ax.scatter(x[0], y[0], z[0], color='green', s=50, label='Start')
    end_marker = ax.scatter(x[-1], y[-1], z[-1], color='red', s=50, label='End')

ax.set_xlabel('TX (mm)')
ax.set_ylabel('TY (mm)')
ax.set_zlabel('TZ (mm)')
# ax.set_title('Flight Path')
# ax.legend()
# Hide x-axis components
# ax.set_yticks([])          # remove ticks
# ax.set_yticklabels([])     # remove tick labels
# ax.yaxis.line.set_color((0.0, 0.0, 0.0, 0.0))  # make axis line transparent

ax.set_xlim(np.min(x), np.max(x))
ax.set_ylim(np.min(y), np.max(y))
ax.set_zlim(np.min(z), np.max(z))
# ax.set_box_aspect([np.ptp(x), np.ptp(y), np.ptp(z)])
# ax.set_aspect('equal')
# --- Range Slider ---
ax_slider = plt.axes([0.25, 0.2, 0.65, 0.03])
slider = RangeSlider(ax_slider, 'Frame Range', 0, len(x) - 2, valinit=(0, len(x) - 2), valstep=1)
print(len(x))

def update(val):
    start, end = int(slider.val[0]), int(slider.val[1] + 1)
    x_trim = x[start:end + 1]
    y_trim = y[start:end + 1]
    z_trim = z[start:end + 1]

    segs, cols = create_colored_line_segments(x_trim, y_trim, z_trim)
    lc.set_segments(segs)
    lc.set_color(cols)

    if show_markers:
        start_marker._offsets3d = ([x_trim[0]], [y_trim[0]], [z_trim[0]])
        end_marker._offsets3d = ([x_trim[-1]], [y_trim[-1]], [z_trim[-1]])

        # Compute min/max
    stats = f"""
X: [{min(x_trim):.3f}, {max(x_trim):.3f}]
Y: [{min(y_trim):.3f}, {max(y_trim):.3f}]
Z: [{min(z_trim):.3f}, {max(z_trim):.3f}]
    """.strip()
    stats_text.set_text(stats)

    fig.canvas.draw_idle()


slider.on_changed(update)

# --- Orthographic/Perspective Toggle ---
ortho_mode = [False]  # Mutable container to keep state


def toggle_projection(event):
    ortho_mode[0] = not ortho_mode[0]
    ax.set_proj_type('ortho' if ortho_mode[0] else 'persp')
    fig.canvas.draw_idle()


ax_button_proj = plt.axes([0.25, 0.13, 0.1, 0.04])
btn_proj = Button(ax_button_proj, 'Toggle Ortho')
btn_proj.on_clicked(toggle_projection)


# --- View Buttons ---
def set_view(elev, azim):
    def inner(event):
        ax.view_init(elev, azim)
        fig.canvas.draw_idle()

    return inner


# --- Fixed View Buttons with Proper Callbacks ---
def make_view_button(label, elev, azim, pos):
    ax_btn = plt.axes(pos)
    btn = Button(ax_btn, label)

    def on_click(event):
        ax.view_init(elev=elev, azim=azim)
        fig.canvas.draw_idle()

    btn.on_clicked(on_click)
    return btn


# Define buttons and positions
btns_info = [
    ('+X', 0, 0, [0.37, 0.05, 0.08, 0.04]),
    ('-X', 0, 180, [0.47, 0.05, 0.08, 0.04]),
    ('+Y', 0, 90, [0.57, 0.05, 0.08, 0.04]),
    ('-Y', 0, -90, [0.37, 0.01, 0.08, 0.04]),
    ('+Z', 90, 0, [0.47, 0.01, 0.08, 0.04]),
    ('-Z', -90, 0, [0.57, 0.01, 0.08, 0.04]),
]

import datetime

def save_figure(event):
    # Get current slice range
    start, end = int(slider.val[0]), int(slider.val[1] + 1)
    x_trim = x[start:end + 1]
    y_trim = y[start:end + 1]
    z_trim = z[start:end + 1]

    # Update line and markers to reflect trimmed path
    segs, cols = create_colored_line_segments(x_trim, y_trim, z_trim)
    lc.set_segments(segs)
    lc.set_color(cols)

    if show_markers:
        start_marker._offsets3d = ([x_trim[0]], [y_trim[0]], [z_trim[0]])
        end_marker._offsets3d = ([x_trim[-1]], [y_trim[-1]], [z_trim[-1]])
    fig.canvas.draw()

    # --- Save current state ---
    prev_facecolor = ax.get_facecolor()
    prev_grid = ax._axis3don  # whether axes are shown
    prev_xlabel, prev_ylabel, prev_zlabel = ax.get_xlabel(), ax.get_ylabel(), ax.get_zlabel()
    prev_xticks, prev_yticks, prev_zticks = ax.get_xticks(), ax.get_yticks(), ax.get_zticks()
    prev_xticklabels = ax.get_xticklabels()
    prev_yticklabels = ax.get_yticklabels()
    prev_zticklabels = ax.get_zticklabels()
    prev_legend = ax.get_legend()

    # --- Hide visual elements ---
    ax.set_facecolor((0, 0, 0, 0))  # transparent background
    ax.set_axis_off()              # turn off grid, axes, and labels
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    if prev_legend:
        prev_legend.remove()

    # Redraw before saving
    fig.canvas.draw()

    # Save only the axes area
    renderer = fig.canvas.get_renderer()
    bbox = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"flight_path_clean_{now}.png"
    fig.savefig(dir + filename, dpi=300, bbox_inches=bbox, transparent=True)
    print(f"Saved clean figure as: {filename}")

    # --- Restore original state ---
    ax.set_facecolor(prev_facecolor)
    ax.set_axis_on()
    ax.set_xlabel(prev_xlabel)
    ax.set_ylabel(prev_ylabel)
    ax.set_zlabel(prev_zlabel)
    ax.set_xticks(prev_xticks)
    ax.set_yticks(prev_yticks)
    ax.set_zticks(prev_zticks)
    ax.set_xticklabels(prev_xticklabels)
    ax.set_yticklabels(prev_yticklabels)
    ax.set_zticklabels(prev_zticklabels)
    if prev_legend:
        ax.legend()  # re-add the legend

    fig.canvas.draw()


def toggle_aspect(event):
    global aspect_equal
    if aspect_equal:
        # Auto aspect (Matplotlib default behavior)
        ax.set_box_aspect(None)
    else:
        # Equal aspect: scale x, y, z equally
        ax.set_box_aspect([np.ptp(x), np.ptp(y), np.ptp(z)])

    aspect_equal = not aspect_equal
    fig.canvas.draw_idle()

# def toggle_aspect(event):
#     global aspect_equal
#     if aspect_equal:
#         ax.set_aspect('auto')
#     else:
#         ax.set_aspect('equal')
#     aspect_equal = not aspect_equal
#     fig.canvas.draw_idle()


view_buttons = [make_view_button(label, elev, azim, pos) for label, elev, azim, pos in btns_info]

# Button placement
ax_btn_save = plt.axes([0.7, 0.13, 0.1, 0.04])
btn_save = Button(ax_btn_save, 'Save Figure')
btn_save.on_clicked(save_figure)

ax_btn_aspect = plt.axes([0.82, 0.13, 0.12, 0.04])
btn_aspect = Button(ax_btn_aspect, 'Toggle Aspect')
btn_aspect.on_clicked(toggle_aspect)

plt.show()
