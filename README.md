# Paddle Quick Inference
* PaddleInference封装的一个快速部署的高层API
* 本项目基于Paddle2.0版本开发

## 高层API特点
* 使用方便快捷
* 代码类似动态图模型的加载和预测
* 将常用的Config和Predictor配置进行封装
* 留出额外的接口以兼容一些特殊的配置操作

## 模型格式
* 目前本项目支持四种推理模型格式，具体请参考下表：

    | 模型计算图 | 模型参数 |
    | -------- | -------- |
    | model | params |
    | \_\_model\_\_ | \* |
    | \_\_model\_\_ | \_\_params\_\_ |
    | \*.pdmodel | \*.pdiparams |

# 安装
* 直接安装
```shell
$ pip install ppqi -i https://pypi.python.org/simple
```
* 通过wheel包进行安装：[下载链接](https://github.com/jm12138/PaddleQuickInference/releases)
```shell
$ pip install [path to whl]
```

## 快速使用
```python
import numpy as np
from ppqi import InferenceModel

# 加载推理模型
model = InferenceModel(
    modelpath=[Inference Model Path], 
    use_gpu=False,
    use_mkldnn=False
)
model.eval()

# 准备数据
inputs = np.random.randn(8, 64, 64, 3).astype(np.float32)

# 前向计算
outputs = model(inputs)
```

## API说明
```python
'''
modelpath：推理模型路径
use_gpu：是否使用GPU进行推理
use_mkldnn：是否使用MKLDNN库进行CPU推理加速

还可以通过InferenceModel.config来对其他选项进行配置
如配置tensorrt：
model.config.enable_tensorrt_engine(
    workspace_size = 1 << 20, 
    max_batch_size = 1, 
    min_subgraph_size = 3, 
    precision_mode=paddle.inference.PrecisionType.Float32, 
    use_static = False, 
    use_calib_mode = False
)
'''
model = InferenceModel(
    modelpath=[Inference Model Path], 
    use_gpu=False,
    use_mkldnn=False
)

'''
将模型设置为推理模式
实际上是使用Config创建Predictor
'''
model.eval()

'''
创建完Predictor后
可打印出模型的输入输出节点的数量和名称
'''
print(model)

'''
根据输入节点的数量和名称准备好数据
数据格式为Ndarray
'''
input_datas = np.random.randn(8, 64, 64, 3).astype(np.float32)

'''
模型前向计算
根据输入节点顺序传入输入数据
batch_size：推理数据批大小
返回结果格式为所有输出节点的输出（Ndarray）
'''
outputs = model(input_datas, batch_size=4)
```

## 部署案例
* [街景动漫化模型 AnimeGAN](./examples/AnimeGAN)
* [人像动漫化模型 UGATIT](./examples/UGATIT)
* [单目深度估计模型 MiDaS](./examples/MiDaS)
* [人脸素描生成 U2Net PortraitGeneration ](./examples/U2Net/PortraitGeneration)
* [人脸检测模型 PyramidBox](./examples/PyramidBox)
* [人像分割模型 SINet PortraitSegmentation](./examples/SINet)
* [人像分割模型 ExtremeC3Net PortraitSegmentation](./examples/ExtremeC3Net)
* [手部关键点检测模型 OpenPose HandsEstimation](./examples/OpenPose/HandsEstimation)
