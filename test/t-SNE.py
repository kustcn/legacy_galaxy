import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from PIL import Image
import numpy as np

save_path='features_target.npy'
# 读取数组数据
data = np.load(save_path, allow_pickle=True)  # 假设数据存储在名为 "data.npy" 的文件中
features = data[:, 0]  # 特征向量
labels = data[:, 1]  # 特征
image_paths = data[:, 2]  # 图片路径
replacement = "test"
new_array = [(replacement+path[66:]) for tup in image_paths for path in tup]

features_array = np.array([np.array(x) for x in features])
features_array = features_array.squeeze()
# 使用 t-SNE 进行降维
tsne = TSNE(n_components=2, random_state=42)
embedded = tsne.fit_transform(features_array)
# 设置不同类别的颜色
num_classes = len(np.unique(labels))
color_map = plt.get_cmap('tab10')
label=['Edge-on','Elliptical','Irregular','Error','Round','Cigar','Spiral']
# 创建子图和坐标轴
fig, ax = plt.subplots()
fig.set_size_inches(12, 8)  # 设置宽度为8英寸，高度为6英寸

# 绘制 t-SNE 可视化结果，并按类别着色
scatter = ax.scatter(embedded[:, 0], embedded[:, 1], c=labels, cmap=color_map)

# 设置散点的大小
sizes = [5] * len(embedded)  # 设置所有散点的初始大小
scatter.set_sizes(sizes)
# cbar = plt.colorbar(scatter)

# 添加图例
handles, _ = scatter.legend_elements()
labels = np.unique(labels)
legend = ax.legend(handles, label, loc="upper right", title="Classes")

legend.get_title().set_fontsize(12)  # 设置标题文字大小
for text in legend.get_texts():
    text.set_fontsize(11.5)  # 设置图例项文字大小


def on_click(event):
    if event.inaxes == ax:
        # 获取鼠标点击位置的坐标
        x, y = event.xdata, event.ydata

        # 计算鼠标点击位置与每个数据点之间的距离
        distances = np.sqrt((x - x_data)**2 + (y - y_data)**2)

        # 获取距离最近的数据点的索引
        closest_index = np.argmin(distances)

        # 获取最近数据点的图像路径
        image_path = new_array[closest_index]
        image = Image.open(image_path)
        # 显示原始图像
        plt.figure()
        plt.imshow(image)
        plt.axis("off")
        plt.show()
# 获取散点图的数据坐标
scatter_data = scatter.get_offsets()
x_data = scatter_data[:, 0]
y_data = scatter_data[:, 1]

# 将点击事件绑定到散点图对象
scatter.figure.canvas.mpl_connect("button_press_event", on_click)
plt.savefig('t-S.png')

# 显示图形
plt.show()
