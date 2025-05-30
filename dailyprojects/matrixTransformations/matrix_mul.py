import numpy as np
import matplotlib.pyplot as plt

# ---------- 1. build example matrices ----------
A = np.array([[2, 1],
              [0, 3],
              [1, 4]])           # 3×2
B = np.array([[ -1, 1],
              [ 2,  -1]])         # 2×2

C = A @ B                        # 3×2 product → three 2-D points

# ---------- 2. define transforms ----------
shift_vec = np.array([2, -1])       # translation
theta = np.radians(45)              # 45° CCW rotation
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

C_shift      = C + shift_vec
C_rot        = (R @ C.T).T
C_shift_rot  = (R @ C_shift.T).T    # shift FIRST, then rotate

# ---------- 3. plotting utility ----------
def scatter(ax, pts, label, marker, color):
    ax.scatter(pts[:, 0], pts[:, 1], marker=marker,
               s=80, facecolors='none' if marker == 'o' else color,
               edgecolors=color, label=label)
    for i, (x, y) in enumerate(pts):
        ax.text(x + 0.1, y + 0.1, f"P{i}", fontsize=8, color=color)

# ---------- 4. make the plot ----------
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal')
ax.grid(ls='--', alpha=0.5)
ax.axhline(0, color='grey', lw=0.8)
ax.axvline(0, color='grey', lw=0.8)

scatter(ax, C,            'C = A @ B',        's', 'tab:blue')
scatter(ax, C_shift,      'Shift (+2, −1)',   'o', 'tab:green')
scatter(ax, C_rot,        'Rotate 45°',       '^', 'tab:red')
scatter(ax, C_shift_rot,  'Shift → Rotate',   'v', 'tab:purple')

ax.set_title('Matrix product and affine transforms')
ax.legend()
plt.show()
