# Paddle Quick Inference
* 基于PaddleInference封装的一个快速部署的高层API

## 高层API特点
* 使用方便快捷
* 代码类似动态图模型的加载和预测
* 将常用的Config和Predictor配置进行封装
* 留出额外的接口以兼容一些特殊的配置操作

## 快速使用
```python
import numpy as np
from inference import InferenceModel

# 加载推理模型
model = InferenceModel(
    modelpath=[Inference Model Path], 
    use_gpu=False,
    use_mkldnn=False,
    combined=False
)
model.eval()

# 准备数据
inputs = np.random.randn(8, 64, 64, 3).astype(np.float32)

# 前向计算
outputs = model(inputs)

# 打印输出
print(outputs.shape)
```
