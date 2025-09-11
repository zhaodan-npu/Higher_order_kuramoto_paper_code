#!/usr/bin/env python3
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

outdir = "results"
k1s = np.linspace(-1, 6.0, 100)
k2s = np.linspace(0, 10.0, 100)
r0s = np.linspace(0.0, 1.0, 200)

# 1. 合并
basin = np.zeros((len(k1s), len(k2s), len(r0s)), dtype=int)
for i in range(len(k1s)):
    data = np.load(f"{outdir}/basin_k1_{i:03d}.npz")
    basin[i, :, :] = data["basin_slice"]

# 2. 保存 TXT
K1g, K2g, R0g = np.meshgrid(k1s, k2s, r0s, indexing='ij')
stack = np.column_stack((
    K1g.ravel(), K2g.ravel(), R0g.ravel(), basin.ravel()
))
np.savetxt(
    f"{outdir}/basin_data.txt", stack,
    header="K1    K2    r0    state(0=inc,1=sync)",
    comments="", fmt="%.6f %.6f %.6f %d"
)

# 3. 画 3D 散点
fig = plt.figure(figsize=(6,5))
ax  = fig.add_subplot(projection="3d")
ax.scatter(
    K1g.ravel(), K2g.ravel(), R0g.ravel(),
    c=basin.ravel(), cmap="bwr",
    s=15, alpha=0.8, edgecolors="k", linewidths=0.2
)
ax.set_xlabel("$K_1$")
ax.set_ylabel("$K_2$")
ax.set_zlabel("$r_0$")
ax.view_init(elev=25, azim=45)
plt.tight_layout()
fig.savefig(f"{outdir}/basin_3d.pdf", dpi=300, bbox_inches="tight")
print("合并并画图完成：", outdir)
