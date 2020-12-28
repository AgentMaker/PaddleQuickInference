# **AnimeGAN的推理部署**
## 在线体验
* AIStudio项目：[PaddleInference：风景图像动漫化模型AnimeGAN的推理部署](https://aistudio.baidu.com/aistudio/projectdetail/1201335)

## 预训练推理模型下载
* 下载链接：[Paddle Quick Inference Examples](https://aistudio.baidu.com/aistudio/datasetdetail/66517)

## 代码示例
```python
# main.py
from inference import InferenceModel
from processor import preprocess, postprocess

# 参数配置
configs = {
    'img_path': 'test.jpg',
    'save_dir': 'save_img',
    'model_name': 'animegan_v1_hayao_60',
    'use_gpu': False,
    'use_mkldnn': False,
    'max_size': 512,
    'min_size': 32
}

# 第一步：数据预处理
input_data = preprocess(
    configs['img_path'], 
    configs['max_size'],  
    configs['min_size']
)

# 第二步：加载模型
model = InferenceModel(
    modelpath=configs['model_name'], 
    use_gpu=configs['use_gpu'], 
    use_mkldnn=configs['use_mkldnn'], 
    combined=False
)
model.eval()

# 第三步：模型推理
output = model(input_data)

# 第四步：结果后处理
postprocess(
    output, 
    configs['save_dir'],
    configs['model_name']
)
```
