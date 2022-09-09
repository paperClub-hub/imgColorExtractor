#
# import torch
# import numpy as np
# import torch.nn as nn
# from sklearn.metrics.pairwise import cosine_similarity
#
# softmax = nn.Softmax()
#
# #设置查询向量和待查询向量
# q = np.array([[1,2,3]])#使用cosine_similarity函数必须为二维向量
# v = np.array([[1,2,3],[4,5,6]])
#
# #计算q和v中每个向量之间的attention得分，此处使用余弦相似度计算，可以采取其他多种计算方式
# sim = cosine_similarity(q,v)
#
# #对计算得到的attention得分进行softmax归一化
# softmax_sim = softmax(torch.tensor(sim[0]))
#
# #依据attention对v中的每一个向量进行加权求和
# attention = 0
# for i in range(v.shape[0]):
# 	attention += v[i] * np.array(softmax_sim[i])
#
# #加权求和后取平均
# attention = attention / v.shape[0]
# attention



from PIL import Image
import colorgramTest

img_path = 'data/test.png'
pil_img = Image.open(img_path)
colors = colorgramTest.extract(pil_img, 10)

print("colors: ", [c.proportion for c in colors])
im = colorgramTest.drawColorBlock(colors, 150)
im.show()



