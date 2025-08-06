import os
import json
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def euler_to_rotation_matrix(roll, pitch, yaw):
    # Convert Euler angles (radians) to rotation matrix (XYZ intrinsic)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx


class VisualizerApp:
    def __init__(self, root, directory):
        self.root = root
        self.directory = directory
        self._is_playing = True
        self.orientations = None

        # Load JSON log file
        json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
        if not json_files:
            raise FileNotFoundError("No JSON log file found in directory")
        json_path = os.path.join(directory, json_files[0])
        with open(json_path, 'r') as f:
            self.log_data = json.load(f)["frames"]

        # Extract frame data from JSON
        self.frames = {}
        timestamps = []
        for entry in self.log_data:
            fid = entry["frame_id"]
            self.frames[fid] = entry
            timestamps.append(entry["time"])
        self.frame_ids = sorted(self.frames.keys())
        self.N = len(self.frame_ids)
        self.timestamps = [self.frames[fid]["time"] for fid in self.frame_ids]

        # Determine frame intervals and frame rate
        if self.N > 1:
            intervals = np.diff(self.timestamps)
            median_interval = np.median(intervals)
        else:
            median_interval = 0.1  # default 10 fps
        self.interval_ms = max(1, int(median_interval))  # minimum 1 ms
        print(f"Determined frame interval: {self.interval_ms} ms")

        # Load images for frames: 'frame_[id].png'
        self.images = {}
        for fid in self.frame_ids:
            img_path = os.path.join(directory, f"frame_{fid}.png")
            if os.path.isfile(img_path):
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.images[fid] = img
            else:
                self.images[fid] = None

        # Extract positions for plotting
        self.positions = np.array([self.frames[fid]["tvec"] for fid in self.frame_ids])

        if "yaw_pitch_roll" in self.frames[fid]:
            self.orientations = np.array([self.frames[fid]["yaw_pitch_roll"] for fid in self.frame_ids])

        # Setup matplotlib figure and axes
        self.fig = plt.figure(figsize=(14, 9))
        # 3D pose visualizer
        self.ax_3d = self.fig.add_subplot(221, projection='3d')
        self.ax_3d.set_title('3D Pose Visualization')
        self.ax_3d.set_xlabel('X (m)')
        self.ax_3d.set_ylabel('Y (m)')
        self.ax_3d.set_zlabel('Z (m)')
        self.ax_3d.view_init(elev=20, azim=45)

        # Video playback
        self.ax_img = self.fig.add_subplot(222)
        self.ax_img.axis('off')
        self.ax_img.set_title('Video Frame')

        # Position plot X,Y,Z vs frame id
        self.ax_pos = self.fig.add_subplot(223)
        self.ax_pos.set_title('Position (m) vs Frame ID')
        self.ax_pos.set_xlabel('Frame ID')
        self.ax_pos.set_ylabel('Position (m)')
        self.pos_lines = {}
        colors = ['r', 'g', 'b']
        for i, c in enumerate(['X', 'Y', 'Z']):
            line, = self.ax_pos.plot(self.frame_ids, self.positions[:, i], label=c, color=colors[i])
            self.pos_lines[c] = line
        self.ax_pos.legend()
        self.cursor_pos = self.ax_pos.axvline(self.frame_ids[0], color='k', linestyle='--')
        self.text_pos = self.ax_pos.text(0.02, 0.95, '', transform=self.ax_pos.transAxes, va='top')

        # Orientation plot roll,pitch,yaw vs frame id
        self.ax_ori = self.fig.add_subplot(224)
        self.ax_ori.set_title('Orientation (rad) vs Frame ID')
        self.ax_ori.set_xlabel('Frame ID')
        self.ax_ori.set_ylabel('Orientation (rad)')
        self.ori_lines = {}
        labels = ['Roll', 'Pitch', 'Yaw']
        if self.orientations is not None:
            for i, c in enumerate(labels):
                line, = self.ax_ori.plot(self.frame_ids, self.orientations[:, i], label=c, color=colors[i])
                self.ori_lines[c] = line
        self.ax_ori.legend()
        self.cursor_ori = self.ax_ori.axvline(self.frame_ids[0], color='k', linestyle='--')
        self.text_ori = self.ax_ori.text(0.02, 0.95, '', transform=self.ax_ori.transAxes, va='top')

        # Set axis limits for 3D plot with padding
        pos_min = np.min(self.positions, axis=0)
        pos_max = np.max(self.positions, axis=0)
        padding = (pos_max - pos_min) * 0.1
        if np.all(padding == 0):
            padding = np.array([0.1, 0.1, 0.1])
        self.ax_3d.set_xlim(pos_min[0] - padding[0], pos_max[0] + padding[0])
        self.ax_3d.set_ylim(pos_min[1] - padding[1], pos_max[1] + padding[1])
        self.ax_3d.set_zlim(pos_min[2] - padding[2], pos_max[2] + padding[2])

        # Initialize 3D pose axes lines (x-red, y-green, z-blue)
        self.pose_axis_length = np.linalg.norm(pos_max - pos_min) * 0.2
        if self.pose_axis_length == 0:
            self.pose_axis_length = 0.1
        self.x_line, = self.ax_3d.plot([], [], [], 'r-', lw=3, label='X axis')
        self.y_line, = self.ax_3d.plot([], [], [], 'g-', lw=3, label='Y axis')
        self.z_line, = self.ax_3d.plot([], [], [], 'b-', lw=3, label='Z axis')
        self.ax_3d.legend()
        self.ax_3d.set_aspect('equal')

        # Image plot init
        self.img_plot = self.ax_img.imshow(np.zeros((240, 320, 3), dtype=np.uint8))

        # Setup slider and buttons
        ax_slider = self.fig.add_axes([0.2, 0.01, 0.6, 0.03])
        self.slider = Slider(ax_slider, 'Frame', 0, self.N - 1, valinit=0, valstep=1)
        self.slider.on_changed(self.on_slider)

        ax_play = self.fig.add_axes([0.05, 0.01, 0.1, 0.04])
        self.btn_play = Button(ax_play, 'Play')
        self.btn_play.on_clicked(self.on_play)

        ax_pause = self.fig.add_axes([0.87, 0.01, 0.1, 0.04])
        self.btn_pause = Button(ax_pause, 'Pause')
        self.btn_pause.on_clicked(self.on_pause)

        # Embed matplotlib canvas in Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Start animation loop
        self.current_index = 0
        self.update_frame()  # initial draw
        self.root.after(self.interval_ms, self.update_loop)

    def on_play(self, event):
        self._is_playing = True

    def on_pause(self, event):
        self._is_playing = False

    def on_slider(self, val):
        self._is_playing = False
        self.current_index = int(val)
        self.update_frame()

    def update_loop(self):
        if self._is_playing:
            self.current_index = (self.current_index + 1) % self.N
            self.slider.eventson = False
            self.slider.set_val(self.current_index)
            self.slider.eventson = True
            self.update_frame()
        self.root.after(self.interval_ms, self.update_loop)

    def update_frame(self):
        fid = self.frame_ids[self.current_index]

        # Update 3D pose visualization
        frame = self.frames.get(fid, None)
        if frame is not None:
            p = np.array(frame["tvec"])
            if self.orientations is not None:
                o = frame["yaw_pitch_roll"]
                R = euler_to_rotation_matrix(*o)
            else:
                R = euler_to_rotation_matrix(0, 0, 0)
            L = self.pose_axis_length

            # X axis (red)
            x_end = p + R[:, 0] * L
            self.x_line.set_data([p[0], x_end[0]], [p[1], x_end[1]])
            self.x_line.set_3d_properties([p[2], x_end[2]])
            # Y axis (green)
            y_end = p + R[:, 1] * L
            self.y_line.set_data([p[0], y_end[0]], [p[1], y_end[1]])
            self.y_line.set_3d_properties([p[2], y_end[2]])
            # Z axis (blue)
            z_end = p + R[:, 2] * L
            self.z_line.set_data([p[0], z_end[0]], [p[1], z_end[1]])
            self.z_line.set_3d_properties([p[2], z_end[2]])
        else:
            self.x_line.set_data([], [])
            self.x_line.set_3d_properties([])
            self.y_line.set_data([], [])
            self.y_line.set_3d_properties([])
            self.z_line.set_data([], [])
            self.z_line.set_3d_properties([])

        # Update image playback: show closest smaller frame id if missing
        img = None
        if fid in self.images and self.images[fid] is not None:
            img = self.images[fid]
        else:
            # find closest smaller frame with image
            idx = self.current_index
            while idx >= 0:
                ftmp = self.frame_ids[idx]
                if self.images.get(ftmp) is not None:
                    img = self.images[ftmp]
                    break
                idx -= 1
        if img is not None:
            self.img_plot.set_data(img)
            # self.ax_img.set_xlim(0, img.shape[1])
            # self.ax_img.set_ylim(0, img.shape[0])
        else:
            # Blank image if none available
            blank = np.zeros((240, 320, 3), dtype=np.uint8)
            self.img_plot.set_data(blank)

        # Update position cursor line & text
        x = self.frame_ids[self.current_index]
        self.cursor_pos.set_xdata([x])
        pos = self.positions[self.current_index]
        self.text_pos.set_text(f'X={pos[0]:.3f}, Y={pos[1]:.3f}, Z={pos[2]:.3f}')

        # Update orientation cursor line & text
        if self.orientations is not None:
            self.cursor_ori.set_xdata([x])
            ori = self.orientations[self.current_index]
            self.text_ori.set_text(f'Roll={ori[0]:.3f}, Pitch={ori[1]:.3f}, Yaw={ori[2]:.3f}')

        self.canvas.draw_idle()


def main():
    root = tk.Tk()
    root.title("3D Pose and Video Viewer")

    # folder = filedialog.askdirectory(title="Select directory with JSON and frames")
    folder = "vicon"
    if not folder:
        print("No folder selected, exiting.")
        return

    app = VisualizerApp(root, folder)
    root.mainloop()


if __name__ == "__main__":
    main()
