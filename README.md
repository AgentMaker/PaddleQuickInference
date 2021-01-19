# Paddle Quick Inference
* Paddle Inference 封装的一个快速部署的高层 API
* 本项目基于 Paddle 2.0 版本开发

## 高层 API 特点
* 使用方便快捷
* 代码类似动态图模型的加载和预测
* 将常用的 Config 和 Predictor 配置进行封装
* 留出额外的接口以兼容一些特殊的配置操作

## 模型格式
* 目前本项目支持四种推理模型格式，具体请参考下表：

    | 模型计算图 | 模型参数 |
    | -------- | -------- |
    | \_\_model\_\_ | \* |
    | model | params |
    | \_\_model\_\_ | \_\_params\_\_ |
    | \*.pdmodel | \*.pdiparams |

# 安装
* 直接安装
```shell
$ pip install ppqi -i https://pypi.python.org/simple
```
* 通过 wheel 包进行安装：[下载链接](https://github.com/jm12138/PaddleQuickInference/releases)
```shell
$ pip install [path to whl]
```

## 快速使用
```python
import numpy as np
from ppqi import InferenceModel

# 加载推理模型
model = InferenceModel([Inference Model Path])
model.eval()

# 准备数据
inputs = np.random.randn(8, 64, 64, 3).astype(np.float32)

# 前向计算
outputs = model(inputs)
```

## API 说明
```python
'''
modelpath：推理模型路径
use_gpu：是否使用 GPU 进行推理
gpu_id：设置使用的 GPU ID
use_mkldnn：是否使用 MKLDNN 库进行 CPU 推理加速
cpu_threads：设置计算库的所使用 CPU 线程数

还可以通过 InferenceModel.config 来对其他选项进行配置
如配置 TensorRT：
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
    gpu_id=0,
    use_mkldnn=False,
    cpu_threads=1
)

'''
将模型设置为推理模式
实际上是使用 Config 创建 Predictor
'''
model.eval()

'''
创建完 Predictor 后
可打印出模型的输入输出节点的数量和名称
'''
print(model)

'''
根据输入节点的数量和名称准备好数据
数据格式为 Ndarray
'''
input_datas = np.random.randn(8, 64, 64, 3).astype(np.float32)

'''
模型前向计算
根据输入节点顺序传入输入数据
batch_size：推理数据批大小
返回结果格式为所有输出节点的输出
数据格式为 Ndarray
'''
outputs = model(input_datas, batch_size=4)
```

## 部署案例
* [街景动漫化模型 AnimeGAN](./examples/AnimeGAN)
* [人像动漫化模型 UGATIT](./examples/UGATIT)
* [单目深度估计模型 MiDaS](./examples/MiDaS)
* [人脸素描生成 U2Net Portrait Generation ](./examples/U2Net/PortraitGeneration)
* [人脸检测模型 Pyramid Box](./examples/PyramidBox)
* [人像分割模型 SINet Portrait Segmentation](./examples/SINet)
* [人像分割模型 ExtremeC3Net Portrait Segmentation](./examples/ExtremeC3Net)
* [手部关键点检测模型 OpenPose Hands Estimation](./examples/OpenPose/HandsEstimation)
