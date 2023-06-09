# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
from aa.inceptionv4 import  inceptionv4
from torchvision import transforms, datasets
from models.han_dcn import resnet_version4_dcn

import numpy as np


# 加载测试数据
data_transform = transforms.Compose([
            transforms.CenterCrop(180),
            transforms.Resize(112),

            transforms.ToTensor(),
            transforms.Normalize([0.18334192, 0.17221707, 0.16791163],[0.15241465, 0.13768229, 0.12769352])])                #七分类
                                     

 
image_vapath_folder=r"/mnt/storage-ssd/luwei/new_dataset/dataset_legacy_7class_5.14/test"
validate_dataset = datasets.ImageFolder(root=image_vapath_folder,#预处理
                                        transform=data_transform)
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,#图片加载
                                                batch_size=1, shuffle=False,#不洗牌
                                                num_workers=8)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")#优先使用Gpu0如果有 没有则cpu


# 设置设备类型
model = resnet_version4_dcn()
ck=torch.load("/mnt/storage-ssd/luwei/gz2_dcn_final/ck_5.20/lam_2/checkpoint99.pth",map_location='cpu')
model.to(device)
# model.load_state_dict(ck['model'])


# %%

#%%
#计算参数量
def count_parameters(mdel):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
#  打印模型参数量
total_params = count_parameters(model)
print(f"Total trainable parameters: {total_params}")



# 计算每一层的参数数量
total_params = 0
for name, parameter in model.named_parameters():
    if not parameter.requires_grad:
        continue
    param_params = parameter.numel()
    print(f"{name}: {param_params}")
    total_params += param_params

print(f"Total number of parameters: {total_params}")

# %%
true_labels = ['Edge-on', 'Elliptical', 'Irregular','Error','Round','Cigar','Spiral']

predicted_labels =['Edge-on', 'Elliptical', 'Irregular','Error','Round','Cigar','Spiral']

# %%
model.eval()
labels=[]
# 对测试集进行预测
y_pred = []
with torch.no_grad():
    for i,(img,label) in enumerate(validate_loader):
          img=img.to(device)
          label=label.to(device)
          outputs = model(img)
          _, predicted = torch.max(outputs.data, 1)
          y_pred.append(predicted.item())
          labels.append(label.item())
cm = confusion_matrix(labels, y_pred)


# %%

# 可视化混淆矩阵
plt.figure(figsize=(12, 12))
plt.imshow(cm, cmap=plt.cm.Blues)
# plt.title('Confusion Matrix')
plt.colorbar(shrink=0.75)
tick_marks = np.arange(len(np.unique(labels)))
plt.xticks(tick_marks, true_labels, fontsize=13)
plt.yticks(tick_marks, predicted_labels, fontsize=13)
plt.xlabel('Predicted Labels', fontsize=14)
plt.ylabel('True Labels', fontsize=14)
matrix = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

thresh = matrix.max() /  2.
for i, j in [(i, j) for i in range(cm.shape[0]) for j in range(cm.shape[1])]:

    plt.text(j, i, f"{cm[i, j]:d}\n{matrix[i, j]:.3%}", ha="center", va="center", color="white" 
             if matrix[i, j] > thresh else "black", fontsize=11.5)
    

plt.tight_layout()

# 保存为PDF文件
plt.savefig('confusion_matridx.pdf',  bbox_inches='tight', pad_inches=0)

# 显示混淆矩阵
plt.show()

#%%






# %%
from sklearn.metrics import classification_report
# 计算分类报告
# y_true = np.concatenate(labels)
# y_pred = np.concatenate(y_pred)
report = classification_report(labels, y_pred, digits=3)

print(report)
# %%
from sklearn.metrics import classification_report, accuracy_score

report = classification_report(labels, y_pred, digits=3, output_dict=True)
accuracy = accuracy_score(labels, y_pred)

average_accuracy = accuracy
print("Average Accuracy:", average_accuracy)


