# Spectral Clustering Demo

基于深度神经网络的谱聚类（Spectral Clustering）演示项目。本仓库实现了 SpectralNet 方法：先用 Siamese 网络学习数据表示，再通过谱聚类在嵌入空间中发现簇结构。

参考论文：[SpectralNet: Spectral Clustering using Deep Neural Networks](https://openreview.net/pdf?id=HJ_aoCyRZ)

![nested C dataset clustering example](https://user-images.githubusercontent.com/9156971/34493923-1abbabe8-efbc-11e7-8788-66c62bc91f4d.png)

## 功能概览

- 在 MNIST、Reuters、嵌套 C 数据集上运行谱聚类
- 支持半监督与带噪声场景
- 可扩展至自定义数据集

## 环境要求

- Python 3.x
- TensorFlow 1.15
- Keras 2.3

安装依赖：

```bash
pip install -r requirements.txt
```

macOS 下载 Reuters 数据集还需安装 `wget`：

```bash
brew install wget
```

## 快速开始

### 内置数据集

```bash
cd src/applications
python run.py --gpu=0 --dset=mnist
```

可选数据集：`mnist` | `reuters` | `cc` | `cc_semisup`

### Reuters 数据预处理

```bash
cd data/reuters
./get_data.sh
python make_reuters.py
```

### 自定义数据集

向 `src/core/data.py` 中的 `get_data` 传入 `(x_train, x_test, y_train, y_test)`，配置超参数后调用 `spectralnet.run()`。完整示例见下方。

```python
import sys
sys.path.insert(0, 'src/')

from applications.spectralnet import run_net
from core.data import get_data

params = {
    'dset': 'new_dataset',
    'val_set_fraction': 0.1,
    'siam_batch_size': 128,
    'n_clusters': 10,
    'affinity': 'nearest_neighbors',
    'n_nbrs': 5,
    'scale_nbrs': True,
    'siam_k': 3,
    'siam_ne': 100,
    'spec_ne': 100,
    'siam_lr': 1e-3,
    'spec_lr': 1e-3,
    'siam_patience': 10,
    'spec_patience': 10,
    'siam_drop': 0.1,
    'spec_drop': 0.1,
    'batch_size': 256,
    'siam_reg': 1e-3,
    'spec_reg': 1e-3,
    'siam_n': 3,
    'siamese_tot_pairs': 100000,
    'arch': [
        {'type': 'relu', 'size': 512},
        {'type': 'relu', 'size': 256},
        {'type': 'relu', 'size': 128},
    ],
    'use_approx': True,
}

x_train, x_test, y_train, y_test = load_new_dataset_data()
data = get_data(params, (x_train, x_test, y_train, y_test))
x_spectralnet, y_spectralnet = run_net(data, params)
```

超参数说明见 `src/applications/run.py`。

## 项目结构

```
spectral-clustering-demo/
├── data/reuters/          # Reuters 数据集脚本
├── src/
│   ├── applications/      # 入口与运行脚本
│   ├── core/              # 网络、训练与数据处理
│   └── pretrain_weights/  # 预训练权重配置
├── requirements.txt
└── README.md
```

## 许可证

MIT License，详见 [LICENSE](LICENSE)。
