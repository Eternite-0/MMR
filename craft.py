import matplotlib.pyplot as plt
import numpy as np

# --- 1. 准备数据 ---
categories = ['粤西地区', '珠三角地区', '广东省平均', '国家规划目标 (2025年)']
# 数据 for '每万人拥有精神科执业医生数 (名)'
doctors = [0.24, 0.58, 0.49, 0.68]
# 数据 for '每万人拥有持证心理咨询师数 (估算, 名)'
counselors = [1.10, 4.50, 3.20, 4.80]

# --- 2. 设置样式和颜色 ---
background_color = '#1A2E44'  # 深蓝色背景
bar_color_doctor = '#4A8AF2'  # 医生数量的蓝色
bar_color_counselor = '#62A7F7' # 咨询师数量的较浅蓝色
text_color = '#FFFFFF'       # 白色文字
spine_color = '#666666'      # 坐标轴颜色

# 设置中文字体（请根据您的操作系统选择合适的字体）
plt.rcParams['font.sans-serif'] = ['SimHei'] # 例如 'SimHei', 'Microsoft YaHei', 'PingFang SC'
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

# --- 3. 创建图表 ---
fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor(background_color)
ax.set_facecolor(background_color)

# 设置X轴位置
x = np.arange(len(categories))
width = 0.35 # 条形的宽度

# 绘制条形图
rects1 = ax.bar(x - width/2, doctors, width, label='每万人拥有精神科执业医生数 (名)', color=bar_color_doctor)
rects2 = ax.bar(x + width/2, counselors, width, label='每万人拥有持证心理咨询师数 (估算, 名)', color=bar_color_counselor)

# --- 4. 美化图表细节 ---
# 添加标题
ax.set_title('每万人拥有心理健康服务者数量对比', color=text_color, fontsize=16, pad=20)

# 设置X轴和Y轴标签
ax.set_xlabel('地区', color=text_color, fontsize=12, labelpad=10)
ax.set_ylabel('数量 (名)', color=text_color, fontsize=12, labelpad=10)

# 设置X轴刻度标签
ax.set_xticks(x)
ax.set_xticklabels(categories, color=text_color, fontsize=10)

# 设置Y轴刻度
ax.tick_params(axis='y', colors=text_color, labelsize=10)
ax.set_ylim(0, 5) # 设置Y轴上限，与您的图片一致

# 隐藏顶部和右侧的边框线
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color(spine_color)
ax.spines['bottom'].set_color(spine_color)

# 设置网格线（可选，但您的原图没有明显网格）
# ax.grid(axis='y', linestyle='--', alpha=0.3, color=spine_color)

# 在条形图顶部添加数据标签
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}', # 格式化为两位小数
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), # 垂直偏移3点
                    textcoords="offset points",
                    ha='center', va='bottom',
                    color=text_color, fontsize=9)

autolabel(rects1)
autolabel(rects2)

# 添加图例，并去除白色边框 (frameon=False)
legend = ax.legend(loc='upper left', facecolor=background_color, edgecolor='none', labelcolor=text_color, fontsize=10, frameon=False)


# --- 5. 调整布局并显示/保存 ---
fig.tight_layout()
# plt.savefig('mental_health_services_chart_no_frame.png', dpi=300, facecolor=background_color) # 取消注释可以保存图片
plt.show()