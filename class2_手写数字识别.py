import paddle
from paddle.nn import Linear
import paddle.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

class MNIST(paddle.nn.Layer):
    def __init__(self):
        super(MNIST, self).__init__()
        self.fc = paddle.nn.Linear(in_features= 28*28, out_features = 1)
    def forward(self, inputs):
        ouputs = self.fc(inputs)
        return ouputs

# 对图像进行归一化，原图像为0-255，现在为0-1
def norm_img(data):
    # 检测实验数据是否正确，正确为[patch_size, 28, 28]
    assert len(data.shape) == 3
    size, x_size, y_size = data.shape[0], data.shape[1], data.shape[2]
    # 归一化
    data = data / 255
    # 重构其矩阵形状
    data = paddle.reshape(data, [size, x_size*y_size])
    return data

# 确保从paddle.vision.datasets.MNIST中加载的图像数据是np.ndarray类型
paddle.vision.set_image_backend('cv2')

def train(model):
    # 启动训练模式
    model.train()
    # 从paddle.vision.datasets.MNIST中加载数据集，一个batch有16,每次打乱数据集
    train_loader = paddle.io.DataLoader(paddle.vision.datasets.MNIST(mode='train'), batch_size = 16, shuffle = True)
    # 定义优化器, 随机梯度下降法
    opt = paddle.optimizer.SGD(learning_rate = 0.001, parameters = model.parameters())
    EPOCH_NUM = 10
    for epoch_id in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            images = norm_img(data[0]).astype('float32')
            labels = data[1].astype('float32')
            # 前向计算
            predicts = model(images)
            # 计算损失
            loss = F.square_error_cost(predicts, labels)
            aver_loss = paddle.mean(loss)

            if batch_id % 1000 == 0 :
                print("epoch: {}, iter: {}, loss is: {}".format(epoch_id, batch_id, aver_loss.numpy()))
            aver_loss.backward()
            opt.step()
            # 清空之前计算的梯度
            opt.clear_grad()

# 提取图片，改变其成为合适的格式
def load_image(path):
    # 打开，并转换成灰度图
    img = Image.open(path).convert("L")
    img = img.resize((28, 28), Image.ANTIALIAS)
    img = np.array(img).reshape(1, -1).astype(np.float32)

    # 图像归一化，保持和数据集的数据范围一致
    img = 1 - img / 255

    return img


# model =  MNIST()
# train(model) 
# paddle.save(model.state_dict(), "mymodel2")

img_path = "example_0.jpg"
im = Image.open(img_path)
plt.imshow(im)
plt.show()
im = im.convert("L")
print(np.array(im).shape)
# 使用Image.ANTIALIAS来采集图片，即高质量采集图片
im = im.resize((28,28), Image.ANTIALIAS)
print(type(im))
plt.show()

model = MNIST()
model.load_dict(paddle.load("mymodel2"))
img = load_image(img_path)
model.eval()
tensor_img = paddle.to_tensor(img)
# tensor_img = paddle.paddle.vision.datasets.MNIST(mode='test')
# print(tensor_img.images)
predicts = model(np.array(tensor_img.images))
print('result',predicts)
#  预测输出取整，即为预测的数字，打印结果
print("本次预测的数字是", predicts.numpy().astype('int32'))

