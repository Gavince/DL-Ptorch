## Pytorch之线性回归

### 线性回归原理

- **加载数据**

  ```python
  feature = 2
  samples = 1000
  
  # 设置参数
  true_w = [2, -3.4]
  true_b = 4.2
  
  # 生成数据
  data = t.randn(samples, feature, dtype=t.float32)
  labels = true_w[0] * data[:, 0] + true_w[1] * data[:, 1] + true_b
  
  # 数据添加噪声
  labels += t.tensor(np.random.normal(0, 0.01, size = labels.size()), dtype=t.float32)
  
  
  def data_iter(batch_size, features, labels):
      """数据的读取"""
      num_examples = len(features)
      indices = list(range(num_examples))
      # 打乱数据
      random.shuffle(indices)
      
      for i in range(0, num_examples, batch_size):
          # 取出索引值
          j = t.LongTensor(indices[i:min(i + batch_size, num_examples)])
          
          yield features.index_select(0, j), labels.index_select(0, j)
  
  ```

- **定义模型 损失函数和优化**

  ```python
  w = t.tensor(np.random.normal(0, 0.01, (feature, 1)), dtype=t.float32)
  b = t.zeros(1, dtype=t.float32)
  
  w.requires_grad_(requires_grad=True)
  b.requires_grad_(requires_grad=True)
  
  def linreg(X, w, b):
      """线性回归模型"""
      return t.mm(X, w) + b
  
  def squared_loss(y_hat, y):
      """均方误差损失"""
      return ((y_hat - y.view(y_hat.size())) **2)/ 2
  
  def sgd(params, lr, batch_size):
      """
      优化函数（小批量的梯度下降）
      小批量的是计算一个batch中所有数据的Loss,再对一个batch的梯度进行平均
      """
      for param in params:
          param.data -= param.grad * lr / batch_size
  ```

- **训练**

  ```python
  # 定义超参数
  
  lr = 0.03
  
  num_epochs = 5
  
  net = linreg
  loss = squared_loss
  
  for epoch in range(num_epochs):
      
      for X, y in data_iter(batch_size=batch_size, features=data, labels=labels):
          
          l = loss(net(X, w, b), y).sum()
          # BP
          l.backward()
          # 更新参数
          sgd([w, b], lr, batch_size=batch_size)
          # 梯度清零
          w.grad.data.zero_()
          b.grad.data.zero_()
       # 计算损失
      train_l = loss(net(data, w, b), labels)
      print("epoch % d, loss %f"%(epoch+1, train_l.mean().item()))
      
  ```

- 结果

  ```python
  #查看训练参数
  for name, param in net.named_parameters():
      print(name, param)
  ```
  
  ```
  linear.weight Parameter containing:
  tensor([[ 2.0004, -3.3998]], requires_grad=True)
  linear.bias Parameter containing:
  tensor([4.1997], requires_grad=True)
  ```

### Pytorch代码实现

```python
import torch
from torch import nn
import numpy as np
import torch.utils.data as Data
from torch.nn import init

# 使得模型的可复现性
torch.manual_seed(1)
# 设置默认的数据格式
torch.set_default_tensor_type('torch.FloatTensor')

# 1.数据处理
num_inputs = 2
num_examples = 1000

true_w = [2, -3.4]
true_b = 4.2

features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b

# 增加数据的噪声, 模拟真实数据
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

# 2.读取数据
batch_size = 10

dataset = Data.TensorDataset(features, labels)
# 查看数据的存储类型 [*dataset]
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True, num_workers=2)

# 3.定义网络
class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)
    
    def forward(self, x):
        return self.linear(x)
net = LinearNet(feature)

# 初始化网络的参数(有多种初始化的方法,但是一般偏值设置为零)
for name, param in net.named_parameters():
    if name == "linear.weight":
        #权重
        init.normal_(param, mean=0.0, std=0.01)
    else:
        #偏值
        init.constant_(param, val=0.0)

# 4.训练
# 超参数
lr = 0.01
batch_size = 10
num_epochs = 5
# 定义损失
critrion = nn.MSELoss()
# 定义优化函数
optimizer = t.optim.SGD(net.parameters(), lr=lr)

for epoch in range(num_epochs):
    for X, y in data_iter:
        out = net(X)
        loss = critrion(out, y.view(-1, 1))
        # 梯度清零
        optimizer.zero_grad()
        loss.backward()
        # 梯度更新
        optimizer.step()
    print("epcoch:%d, loss:%f"%(epoch, loss.item()))
```
结果:

```
epcoch:0, loss:0.861575
epcoch:1, loss:0.015179
epcoch:2, loss:0.000336
epcoch:3, loss:0.000168
epcoch:4, loss:0.000151
```

### 总结

### 参考




