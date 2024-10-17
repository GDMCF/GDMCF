import torch
from torch.nn import functional as F

# 假设你有以下形状为[a,b]的二值矩阵
matrix = torch.tensor([[1, 0, 1],
                       [0, 1, 0]], dtype=torch.long)  # 例子中的a=2, b=3
print('matrix:', matrix.shape)
# 使用F.one_hot进行转换
# 注意：F.one_hot需要一个LongTensor作为输入，并且默认情况下不会添加一个新的维度，
# 因此我们需要手动指定输出的深度（在这个案例中为2，因为是0和1的one-hot编码）
one_hot_matrix = F.one_hot(matrix, num_classes=2)

# 由于F.one_hot返回的tensor默认情况下不会在最后一个维度展开，而是作为列表的元素，
# 我们需要做进一步处理来得到预期的形状[a,b,2]
#one_hot_matrix = one_hot_matrix.permute(0, 2, 1)  # 这里假设初始矩阵是[a,b]，我们需要转置到[a,2,b]

# 打印转换后的one-hot矩阵
print(one_hot_matrix.shape)  # 应该输出 torch.Size([2, 2, 3])
print(one_hot_matrix)