"""
nuScenes Interactive Visualizer
================================
Full GUI to browse samples across scenes.

Controls
--------
  • Scene dropdown  – jump to any scene
  • Sample slider   – scrub through samples in the scene
  • ← / → buttons  – step one sample at a time
  • Toggle checkboxes – show/hide LiDAR points and GT boxes
  • Depth sliders   – min/max depth clipping
  • Point-size / subsample spinboxes
  • Refresh button  – re-render with current settings
  • Keyboard: Left/Right arrows to navigate, R to refresh

Requirements
------------
  pip install nuscenes-devkit pyquaternion matplotlib Pillow
  (tkinter ships with Python on Windows and most Linux distros)
"""

import tkinter as tk
from tkinter import ttk
import threading

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import numpy as np
from PIL import Image

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion

# ═══════════════════════════════════════════════════════════
#  CONFIG  – only thing you need to change
# ═══════════════════════════════════════════════════════════
DATAROOT = r'D:\teamcarla\futr3d\data\nuscenes\rgb'
VERSION  = 'v1.0-mini'

# ═══════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════
CAM_ORDER = [
    'CAM_FRONT_LEFT', 'CAM_FRONT',      'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT',  'CAM_BACK',       'CAM_BACK_RIGHT',
]
CAM_LABELS = {
    'CAM_FRONT':       'Front',
    'CAM_FRONT_LEFT':  'Front Left',
    'CAM_FRONT_RIGHT': 'Front Right',
    'CAM_BACK':        'Back',
    'CAM_BACK_LEFT':   'Back Left',
    'CAM_BACK_RIGHT':  'Back Right',
}
LIDAR_CHANNEL = 'LIDAR_TOP'
BOX_PALETTE   = [
    '#FF4040', '#FFB020', '#20D9FF', '#A0FF40',
    '#FF40D9', '#40FFA0', '#FF8040', '#40A0FF',
]
BG = '#0d0d0d'

# ═══════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════

def build_cat_colour_map(nusc, sample):
    """Return {category_name: hex_colour} for this sample's annotations."""
    names = sorted({
        nusc.get('category',
            nusc.get('instance',
                nusc.get('sample_annotation', tok)['instance_token']
            )['category_token']
        )['name']
        for tok in sample['anns']
    })
    return {n: BOX_PALETTE[i % len(BOX_PALETTE)] for i, n in enumerate(names)}


def project_lidar(nusc, sample, lidar_sd, cam_sd, cam_intrinsic,
                  min_depth, max_depth, subsample):
    """
    Transform LiDAR → camera frame, project, return (px, py, depths).
    Returns empty arrays if no points visible.
    """
    pc = LidarPointCloud.from_file(
        nusc.get_sample_data_path(sample['data'][LIDAR_CHANNEL])
    )

    # lidar → ego
    lcs = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
    pc.rotate(Quaternion(lcs['rotation']).rotation_matrix)
    pc.translate(np.array(lcs['translation']))

    # ego → global
    lep = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
    pc.rotate(Quaternion(lep['rotation']).rotation_matrix)
    pc.translate(np.array(lep['translation']))

    # global → cam ego
    cep = nusc.get('ego_pose', cam_sd['ego_pose_token'])
    pc.translate(-np.array(cep['translation']))
    pc.rotate(Quaternion(cep['rotation']).rotation_matrix.T)

    # cam ego → camera sensor
    ccs = nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
    pc.translate(-np.array(ccs['translation']))
    pc.rotate(Quaternion(ccs['rotation']).rotation_matrix.T)

    depths = pc.points[2, :]
    pts2d  = view_points(pc.points[:3, :], cam_intrinsic, normalize=True)

    # placeholder image size to get W/H – will be clipped later
    # We use cam_intrinsic to estimate principal point as image centre approx
    cx = cam_intrinsic[0, 2] * 2
    cy = cam_intrinsic[1, 2] * 2

    mask  = (depths > min_depth) & (depths < max_depth)
    mask &= (pts2d[0] > 0) & (pts2d[0] < cx)
    mask &= (pts2d[1] > 0) & (pts2d[1] < cy)

    idx = np.where(mask)[0][::max(1, subsample)]
    return pts2d[0, idx], pts2d[1, idx], depths[idx]


# ═══════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ═══════════════════════════════════════════════════════════

class NuScenesViewer:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title('nuScenes Interactive Viewer')
        root.configure(bg='#1a1a2e')
        root.minsize(1200, 800)

        # ── load dataset ────────────────────────────────────────────────────
        self.status_var = tk.StringVar(value='Loading nuScenes …')
        self._build_ui()
        root.update()

        self.nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)

        # build scene list
        self.scenes = self.nusc.scene                          # list of scene dicts
        scene_names = [f"[{i:02d}] {s['name']} – {s['description'][:45]}"
                       for i, s in enumerate(self.scenes)]
        self.scene_combo['values'] = scene_names
        self.scene_combo.current(0)

        self._load_scene(0)
        self.status_var.set('Ready')

    # ──────────────────────────────────────────────────────────────────────────
    #  UI construction
    # ──────────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = self.root

        # ── top control bar ──────────────────────────────────────────────────
        ctrl = tk.Frame(root, bg='#16213e', pady=6, padx=10)
        ctrl.pack(side=tk.TOP, fill=tk.X)

        # Scene selector
        tk.Label(ctrl, text='Scene:', bg='#16213e', fg='#e0e0e0',
                 font=('Consolas', 10)).pack(side=tk.LEFT, padx=(0, 4))
        self.scene_combo = ttk.Combobox(ctrl, width=52, state='readonly',
                                        font=('Consolas', 9))
        self.scene_combo.pack(side=tk.LEFT, padx=(0, 14))
        self.scene_combo.bind('<<ComboboxSelected>>', self._on_scene_select)

        # Sample navigation
        tk.Label(ctrl, text='Sample:', bg='#16213e', fg='#e0e0e0',
                 font=('Consolas', 10)).pack(side=tk.LEFT)
        self.btn_prev = tk.Button(ctrl, text='◀', command=self._prev_sample,
                                  bg='#0f3460', fg='white', relief='flat',
                                  font=('Consolas', 11), padx=6)
        self.btn_prev.pack(side=tk.LEFT, padx=2)

        self.sample_var = tk.IntVar(value=0)
        self.sample_slider = tk.Scale(
            ctrl, from_=0, to=39, variable=self.sample_var,
            orient=tk.HORIZONTAL, length=220,
            bg='#16213e', fg='#e0e0e0', troughcolor='#0f3460',
            highlightthickness=0, showvalue=True,
            command=self._on_slider,
        )
        self.sample_slider.pack(side=tk.LEFT, padx=2)

        self.btn_next = tk.Button(ctrl, text='▶', command=self._next_sample,
                                  bg='#0f3460', fg='white', relief='flat',
                                  font=('Consolas', 11), padx=6)
        self.btn_next.pack(side=tk.LEFT, padx=2)

        self.sample_label = tk.Label(ctrl, text='0 / 0',
                                     bg='#16213e', fg='#53d8fb',
                                     font=('Consolas', 10), width=8)
        self.sample_label.pack(side=tk.LEFT, padx=6)

        # Refresh button
        tk.Button(ctrl, text='⟳  Refresh', command=self._render,
                  bg='#e94560', fg='white', relief='flat',
                  font=('Consolas', 10, 'bold'), padx=10, pady=2
                  ).pack(side=tk.LEFT, padx=12)

        # ── settings bar ────────────────────────────────────────────────────
        sbar = tk.Frame(root, bg='#1a1a2e', pady=4, padx=10)
        sbar.pack(side=tk.TOP, fill=tk.X)

        def chk(parent, text, var, **kw):
            return tk.Checkbutton(parent, text=text, variable=var,
                                  bg='#1a1a2e', fg='#e0e0e0',
                                  selectcolor='#0f3460',
                                  activebackground='#1a1a2e',
                                  font=('Consolas', 9), **kw)

        self.show_lidar = tk.BooleanVar(value=True)
        self.show_boxes = tk.BooleanVar(value=True)
        self.show_labels = tk.BooleanVar(value=True)
        chk(sbar, 'LiDAR points', self.show_lidar).pack(side=tk.LEFT, padx=6)
        chk(sbar, 'GT boxes',     self.show_boxes).pack(side=tk.LEFT, padx=6)
        chk(sbar, 'Box labels',   self.show_labels).pack(side=tk.LEFT, padx=6)

        sep = tk.Frame(sbar, bg='#333', width=2, height=22)
        sep.pack(side=tk.LEFT, padx=10)

        # Depth sliders
        tk.Label(sbar, text='Depth min:', bg='#1a1a2e', fg='#aaa',
                 font=('Consolas', 9)).pack(side=tk.LEFT)
        self.depth_min = tk.DoubleVar(value=1.0)
        tk.Scale(sbar, from_=0, to=10, resolution=0.5,
                 variable=self.depth_min, orient=tk.HORIZONTAL,
                 length=100, bg='#1a1a2e', fg='#e0e0e0',
                 troughcolor='#0f3460', highlightthickness=0,
                 showvalue=True, font=('Consolas', 8)
                 ).pack(side=tk.LEFT, padx=2)

        tk.Label(sbar, text='max:', bg='#1a1a2e', fg='#aaa',
                 font=('Consolas', 9)).pack(side=tk.LEFT)
        self.depth_max = tk.DoubleVar(value=60.0)
        tk.Scale(sbar, from_=10, to=100, resolution=2.0,
                 variable=self.depth_max, orient=tk.HORIZONTAL,
                 length=120, bg='#1a1a2e', fg='#e0e0e0',
                 troughcolor='#0f3460', highlightthickness=0,
                 showvalue=True, font=('Consolas', 8)
                 ).pack(side=tk.LEFT, padx=2)

        sep2 = tk.Frame(sbar, bg='#333', width=2, height=22)
        sep2.pack(side=tk.LEFT, padx=10)

        tk.Label(sbar, text='Subsample:', bg='#1a1a2e', fg='#aaa',
                 font=('Consolas', 9)).pack(side=tk.LEFT)
        self.subsample_var = tk.IntVar(value=3)
        tk.Spinbox(sbar, from_=1, to=20, textvariable=self.subsample_var,
                   width=3, font=('Consolas', 9), bg='#0f3460', fg='white',
                   buttonbackground='#0f3460', relief='flat'
                   ).pack(side=tk.LEFT, padx=4)

        tk.Label(sbar, text='Pt size:', bg='#1a1a2e', fg='#aaa',
                 font=('Consolas', 9)).pack(side=tk.LEFT, padx=(8, 0))
        self.ptsize_var = tk.IntVar(value=3)
        tk.Spinbox(sbar, from_=1, to=15, textvariable=self.ptsize_var,
                   width=3, font=('Consolas', 9), bg='#0f3460', fg='white',
                   buttonbackground='#0f3460', relief='flat'
                   ).pack(side=tk.LEFT, padx=4)

        # Colormap selector
        tk.Label(sbar, text='Cmap:', bg='#1a1a2e', fg='#aaa',
                 font=('Consolas', 9)).pack(side=tk.LEFT, padx=(8, 0))
        self.cmap_var = tk.StringVar(value='plasma')
        cmap_cb = ttk.Combobox(sbar, textvariable=self.cmap_var,
                               values=['plasma', 'jet', 'viridis', 'turbo',
                                       'inferno', 'magma', 'cool', 'hot'],
                               width=8, state='readonly', font=('Consolas', 9))
        cmap_cb.pack(side=tk.LEFT, padx=4)

        # ── matplotlib canvas ────────────────────────────────────────────────
        canvas_frame = tk.Frame(root, bg=BG)
        canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.fig, self.axes = plt.subplots(2, 3, figsize=(22, 11))
        self.fig.patch.set_facecolor(BG)
        plt.subplots_adjust(left=0.01, right=0.99, top=0.93,
                            bottom=0.04, wspace=0.04, hspace=0.12)

        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar_frame = tk.Frame(canvas_frame, bg=BG)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        NavigationToolbar2Tk(self.canvas, toolbar_frame)

        # ── status bar ───────────────────────────────────────────────────────
        tk.Label(root, textvariable=self.status_var,
                 bg='#0d0d1a', fg='#53d8fb',
                 font=('Consolas', 9), anchor='w', padx=8
                 ).pack(side=tk.BOTTOM, fill=tk.X)

        # ── keyboard shortcuts ───────────────────────────────────────────────
        root.bind('<Left>',  lambda e: self._prev_sample())
        root.bind('<Right>', lambda e: self._next_sample())
        root.bind('<r>',     lambda e: self._render())
        root.bind('<R>',     lambda e: self._render())

    # ──────────────────────────────────────────────────────────────────────────
    #  Scene / sample management
    # ──────────────────────────────────────────────────────────────────────────

    def _load_scene(self, scene_idx: int):
        """Build ordered sample list for the chosen scene."""
        self.scene_idx = scene_idx
        scene = self.scenes[scene_idx]

        # walk linked list
        samples = []
        tok = scene['first_sample_token']
        while tok:
            s = self.nusc.get('sample', tok)
            samples.append(s)
            tok = s['next']
        self.samples = samples

        n = len(samples)
        self.sample_slider.config(to=max(0, n - 1))
        self.sample_var.set(0)
        self.sample_label.config(text=f'1 / {n}')
        self._render()

    def _on_scene_select(self, _event=None):
        idx = self.scene_combo.current()
        self._load_scene(idx)

    def _on_slider(self, _value=None):
        n   = len(self.samples)
        idx = self.sample_var.get()
        self.sample_label.config(text=f'{idx + 1} / {n}')
        self._render()

    def _prev_sample(self):
        v = max(0, self.sample_var.get() - 1)
        self.sample_var.set(v)
        self._on_slider()

    def _next_sample(self):
        v = min(len(self.samples) - 1, self.sample_var.get() + 1)
        self.sample_var.set(v)
        self._on_slider()

    # ──────────────────────────────────────────────────────────────────────────
    #  Rendering
    # ──────────────────────────────────────────────────────────────────────────

    def _render(self):
        """Render current sample (runs in a background thread to keep UI alive)."""
        self.status_var.set('Rendering …')
        self.root.update_idletasks()
        t = threading.Thread(target=self._render_worker, daemon=True)
        t.start()

    def _render_worker(self):
        try:
            self._do_render()
        except Exception as exc:
            self.status_var.set(f'Error: {exc}')
            raise

    def _do_render(self):
        sample     = self.samples[self.sample_var.get()]
        nusc       = self.nusc
        min_d      = self.depth_min.get()
        max_d      = self.depth_max.get()
        sub        = max(1, self.subsample_var.get())
        pt_size    = max(1, self.ptsize_var.get())
        cmap       = self.cmap_var.get()
        do_lidar   = self.show_lidar.get()
        do_boxes   = self.show_boxes.get()
        do_labels  = self.show_labels.get()

        lidar_token = sample['data'][LIDAR_CHANNEL]
        lidar_sd    = nusc.get('sample_data', lidar_token)
        cat_colour  = build_cat_colour_map(nusc, sample)

        axes_flat = self.axes.flatten()

        for ax in axes_flat:
            ax.cla()
            ax.set_facecolor(BG)

        # remove old colorbars
        for attr in list(vars(self).keys()):
            if attr.startswith('_cb_'):
                try:
                    getattr(self, attr).remove()
                except Exception:
                    pass
                delattr(self, attr)

        for ax_i, (ax, cam_channel) in enumerate(zip(axes_flat, CAM_ORDER)):
            cam_token = sample['data'][cam_channel]
            cam_sd    = nusc.get('sample_data', cam_token)

            img = Image.open(nusc.get_sample_data_path(cam_token))
            W, H = img.size

            _, boxes, cam_intrinsic_list = nusc.get_sample_data(cam_token)
            cam_intrinsic = np.array(cam_intrinsic_list)

            ax.imshow(img)
            ax.set_xlim(0, W)
            ax.set_ylim(H, 0)

            # ── GT boxes ─────────────────────────────────────────────────
            if do_boxes:
                for box in boxes:
                    col = cat_colour.get(box.name, '#ffffff')
                    box.render(ax, view=cam_intrinsic, normalize=True,
                               colors=(col, col, col), linewidth=1.8)
                    if do_labels:
                        c2d = view_points(
                            box.center.reshape(3, 1), cam_intrinsic, normalize=True
                        )
                        cx2, cy2 = c2d[0, 0], c2d[1, 0]
                        if 0 < cx2 < W and 0 < cy2 < H:
                            lbl = box.name.split('.')[-1]
                            ax.text(cx2, cy2 - 6, lbl,
                                    color=col, fontsize=6, fontweight='bold',
                                    ha='center', va='bottom',
                                    bbox=dict(boxstyle='round,pad=0.15',
                                              fc='black', ec='none', alpha=0.55))

            # ── LiDAR projection ─────────────────────────────────────────
            if do_lidar:
                px, py, d = project_lidar(
                    nusc, sample, lidar_sd, cam_sd, cam_intrinsic,
                    min_d, max_d, sub
                )
                if len(d):
                    vmin = np.percentile(d, 2)
                    vmax = np.percentile(d, 98)
                    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                    sc = ax.scatter(px, py, c=d, s=pt_size, cmap=cmap,
                                    norm=norm, alpha=0.85, linewidths=0, zorder=2)
                    cb = self.fig.colorbar(sc, ax=ax, fraction=0.025,
                                           pad=0.01, aspect=30)
                    cb.set_label('depth (m)', color='#aaa', fontsize=7)
                    cb.ax.yaxis.set_tick_params(color='#aaa', labelsize=6)
                    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='#aaa')
                    cb.outline.set_edgecolor('#333')
                    setattr(self, f'_cb_{ax_i}', cb)

            ax.set_title(CAM_LABELS.get(cam_channel, cam_channel),
                         color='white', fontsize=10, pad=4)
            ax.axis('off')

        # ── legend ───────────────────────────────────────────────────────────
        if do_boxes and cat_colour:
            handles = [
                plt.Line2D([0], [0], color=col, linewidth=2,
                           label=name.split('.')[-1])
                for name, col in cat_colour.items()
            ]
            self.fig.legend(
                handles=handles,
                loc='lower center',
                ncol=min(len(handles), 9),
                frameon=True, framealpha=0.25,
                facecolor='#1a1a1a', edgecolor='#444',
                labelcolor='white', fontsize=8,
                bbox_to_anchor=(0.5, 0.00),
            )

        scene  = self.scenes[self.scene_idx]
        s_idx  = self.sample_var.get()
        ts     = sample['timestamp']
        self.fig.suptitle(
            f"Scene: {scene['name']}  |  Sample {s_idx + 1}/{len(self.samples)}"
            f"  |  token: {sample['token'][:8]}…  |  ts: {ts}",
            color='white', fontsize=11, fontweight='bold',
        )

        self.canvas.draw_idle()
        self.status_var.set(
            f"Scene {self.scene_idx} · sample {s_idx + 1}/{len(self.samples)}"
            f" · {len(sample['anns'])} annotations"
            f"  |  ← → to navigate   R to refresh"
        )


# ═══════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════

def main():
    root = tk.Tk()

    # dark ttk theme
    style = ttk.Style(root)
    try:
        style.theme_use('clam')
    except Exception:
        pass
    style.configure('TCombobox', fieldbackground='#0f3460', background='#0f3460',
                    foreground='white', selectbackground='#e94560',
                    selectforeground='white')

    app = NuScenesViewer(root)
    root.mainloop()


if __name__ == '__main__':
    main()