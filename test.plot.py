import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

def draw_rounded_rect(ax, x, y, w, h, color, label, fontsize=10, alpha=1.0, zorder=1):
    # 绘制更现代的圆角矩形
    box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.2", 
                                 linewidth=1.5, edgecolor=color, facecolor='white', alpha=alpha, zorder=zorder)
    # 添加轻微的填充色（带透明度）
    fill_box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.2", 
                                      linewidth=0, facecolor=color, alpha=0.1, zorder=zorder-1)
    ax.add_patch(box)
    ax.add_patch(fill_box)
    ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=fontsize, weight='bold', color='#333333', zorder=zorder+1)
    return x+w, y+h/2 # Return connect point

def draw_circle_op(ax, x, y, r, label, color='black'):
    circle = patches.Circle((x, y), r, edgecolor=color, facecolor='white', linewidth=1.5, zorder=10)
    ax.add_patch(circle)
    ax.text(x, y, label, ha='center', va='center', fontsize=12, weight='bold', zorder=11)

# 设置画布
fig, ax = plt.subplots(figsize=(16, 8))
ax.set_xlim(0, 20)
ax.set_ylim(0, 10)
ax.axis('off')

# 颜色定义 (Nature/Science 风格配色)
c_input = "#607d8b"  # 蓝灰
c_norm = "#7986cb"   # 浅靛蓝
c_linear = "#4db6ac" # 蓝绿
c_conv = "#ffb74d"   # 橙色
c_ssm = "#009688"    # 深青色 (Mamba核心)
c_act = "#e57373"    # 浅红
c_gate = "#ba68c8"   # 紫色

# ==========================================
# PART 1: The Macro Architecture (Top Left)
# ==========================================

# Title
ax.text(1, 9.5, "A. Meteo Mamba Architecture (Macro)", fontsize=14, weight='bold', color='#2c3e50')

# Input Grid
ax.add_patch(patches.Rectangle((1, 8), 1, 1, facecolor='#cfd8dc', edgecolor='#455a64', lw=1.5))
ax.text(1.5, 8.5, "Input\n(H,W,C)", ha='center', va='center', fontsize=9)
ax.annotate("", xy=(2.5, 8.5), xytext=(2.1, 8.5), arrowprops=dict(arrowstyle="->", lw=1.5))

# Patch Embed
draw_rounded_rect(ax, 2.5, 8.1, 1.5, 0.8, c_input, "Patch\nEmbed")
ax.annotate("", xy=(4.5, 8.5), xytext=(4.1, 8.5), arrowprops=dict(arrowstyle="->", lw=1.5))

# Mamba Layers (Stacked)
ax.text(4.5, 9.1, "N × Layers", fontsize=9, color='#555')
draw_rounded_rect(ax, 4.5, 7.8, 4.0, 1.4, c_ssm, "")
# Inner blocks to show stacking
draw_rounded_rect(ax, 4.7, 8.1, 1.0, 0.8, c_ssm, "Mamba\nBlock", alpha=0.8)
ax.text(6.2, 8.5, "...", fontsize=12)
draw_rounded_rect(ax, 7.0, 8.1, 1.0, 0.8, c_ssm, "Mamba\nBlock", alpha=0.8)

ax.annotate("", xy=(9, 8.5), xytext=(8.6, 8.5), arrowprops=dict(arrowstyle="->", lw=1.5))

# Prediction Head
draw_rounded_rect(ax, 9, 8.1, 1.5, 0.8, c_linear, "Prediction\nHead")
ax.annotate("", xy=(11, 8.5), xytext=(10.6, 8.5), arrowprops=dict(arrowstyle="->", lw=1.5))

# Output
ax.add_patch(patches.Rectangle((11, 8), 1, 1, facecolor='#cfd8dc', edgecolor='#455a64', lw=1.5))
ax.text(11.5, 8.5, "Forecast\n(H,W,C)", ha='center', va='center', fontsize=9)


# ==========================================
# PART 2: The Mamba Block Detail (Main Focus)
# ==========================================

ax.text(1, 6.5, "B. Mamba Block Details (Micro)", fontsize=14, weight='bold', color='#2c3e50')

# Coordinate system for the block diagram
base_x = 2
base_y = 1
block_w = 12
block_h = 5

# Outer Box (Residual Block)
outer_box = patches.FancyBboxPatch((base_x, base_y), block_w, block_h, boxstyle="round,pad=0.2", 
                                   linewidth=2, edgecolor="#90a4ae", facecolor="#f5f5f5", zorder=0)
ax.add_patch(outer_box)

# Input X
ax.text(base_x - 1, base_y + 2.5, "X (Input)", ha='center', fontsize=11, weight='bold')
ax.annotate("", xy=(base_x, base_y + 2.5), xytext=(base_x - 0.8, base_y + 2.5), arrowprops=dict(arrowstyle="->", lw=2))

# Layer Norm
draw_rounded_rect(ax, base_x + 0.5, base_y + 2.1, 1.2, 0.8, c_norm, "Norm")

# Splitting Point
split_x = base_x + 2.2
split_y = base_y + 2.5
ax.annotate("", xy=(split_x, split_y), xytext=(base_x + 1.8, split_y), arrowprops=dict(arrowstyle="-", lw=2))

# --- Upper Branch (The Main SSM Path) ---
path_up_y = base_y + 3.5
ax.plot([split_x, split_x, split_x+0.5], [split_y, path_up_y, path_up_y], color='black', lw=1.5) # Line up

# Linear Projection
draw_rounded_rect(ax, split_x+0.5, path_up_y-0.4, 1.5, 0.8, c_linear, "Linear")
ax.annotate("", xy=(split_x+2.3, path_up_y), xytext=(split_x+2.1, path_up_y), arrowprops=dict(arrowstyle="->", lw=1.5))

# Conv1d
draw_rounded_rect(ax, split_x+2.3, path_up_y-0.4, 1.5, 0.8, c_conv, "Conv1d")
ax.annotate("", xy=(split_x+4.1, path_up_y), xytext=(split_x+3.9, path_up_y), arrowprops=dict(arrowstyle="->", lw=1.5))

# SiLU
draw_rounded_rect(ax, split_x+4.1, path_up_y-0.4, 1.2, 0.8, c_act, "SiLU")
ax.annotate("", xy=(split_x+5.6, path_up_y), xytext=(split_x+5.4, path_up_y), arrowprops=dict(arrowstyle="->", lw=1.5))

# SSM (The Core)
draw_rounded_rect(ax, split_x+5.6, path_up_y-0.6, 2.0, 1.2, c_ssm, "SSM\n(Discretize,\nSelect)", fontsize=9)
ax.annotate("", xy=(split_x+8.0, path_up_y), xytext=(split_x+7.7, path_up_y), arrowprops=dict(arrowstyle="->", lw=1.5))

# --- Lower Branch (The Gating Path) ---
path_low_y = base_y + 1.0
ax.plot([split_x, split_x, split_x+0.5], [split_y, path_low_y, path_low_y], color='black', lw=1.5) # Line down

# Linear Projection
draw_rounded_rect(ax, split_x+0.5, path_low_y-0.4, 1.5, 0.8, c_linear, "Linear")
ax.annotate("", xy=(split_x+4.1, path_low_y), xytext=(split_x+2.1, path_low_y), arrowprops=dict(arrowstyle="->", lw=1.5))

# SiLU (Activation for Gating)
draw_rounded_rect(ax, split_x+4.1, path_low_y-0.4, 1.2, 0.8, c_act, "SiLU")

# --- Merging (Element-wise Multiplication) ---
merge_x = split_x + 8.5
draw_circle_op(ax, merge_x, path_up_y, 0.3, "×") # The multiplication happens at upper level logically or middle

# Connecting Lower branch to Multiply
# Custom path: from lower SiLU to Multiply node
ax.plot([split_x+5.4, merge_x, merge_x], [path_low_y, path_low_y, path_up_y-0.3], color='black', lw=1.5, linestyle='--') 
ax.plot([split_x+7.7, merge_x], [path_up_y, path_up_y], color='black', lw=1.5) # From SSM to Mult

# Output Linear
ax.annotate("", xy=(merge_x+1.0, path_up_y), xytext=(merge_x+0.3, path_up_y), arrowprops=dict(arrowstyle="->", lw=1.5))
draw_rounded_rect(ax, merge_x+1.0, path_up_y-0.4, 1.5, 0.8, c_linear, "Linear")

# Residual Connection
# From Input (before Norm) to Output
res_y = base_y + 4.5
ax.plot([base_x - 0.5, base_x + 11, base_x + 11], [base_y + 2.5, res_y, path_up_y + 0.5], color='#777', lw=1.5, linestyle='-.')
draw_circle_op(ax, base_x + 11, path_up_y, 0.3, "+")

# Final Output Arrow
ax.annotate("", xy=(base_x + 12.5, path_up_y), xytext=(base_x + 11.3, path_up_y), arrowprops=dict(arrowstyle="->", lw=2))
ax.text(base_x + 12.6, path_up_y, "Output", va='center', fontsize=11, weight='bold')

plt.tight_layout()
plt.show()