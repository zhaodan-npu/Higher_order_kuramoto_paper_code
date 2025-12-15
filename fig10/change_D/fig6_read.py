import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 全局字体设成 Times New Roman
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12,          # 基础字号
})

# 这里只用两张图
img_paths = [
    "fig_a_amp_heatmap_Dalpha.png",
    "fig_b_spike_heatmap_Dalpha.png",
]

labels = ['(a)', '(b)']

# 1 行 2 列
fig, axes = plt.subplots(1, 2, figsize=(6, 3.5))

for ax, img_path, label in zip(axes.flat, img_paths, labels):
    img = mpimg.imread(img_path)
    ax.imshow(img)
    ax.axis('off')  # 不显示坐标轴

    # 左上角标注 (a)、(b)
    ax.text(
        -0.1, 0.98, label,
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=10,
        fontweight='normal',
    )

plt.tight_layout()

plt.savefig("combined_with_labels_1x2.pdf", bbox_inches='tight')
plt.savefig("combined_with_labels_1x2.png", dpi=600, bbox_inches='tight')

plt.close(fig)
print("Saved: combined_with_labels_1x2.pdf and combined_with_labels_1x2.png")
