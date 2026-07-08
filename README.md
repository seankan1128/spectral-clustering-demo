# Spectral Clustering Demo

A demonstration project for deep spectral clustering using neural networks. This repository implements the SpectralNet approach: a Siamese network first learns data representations, then spectral clustering discovers cluster structure in the embedding space.

Paper reference: [SpectralNet: Spectral Clustering using Deep Neural Networks](https://openreview.net/pdf?id=HJ_aoCyRZ)

![nested C dataset clustering example](https://user-images.githubusercontent.com/9156971/34493923-1abbabe8-efbc-11e7-8788-66c62bc91f4d.png)

## Features

- Run spectral clustering on MNIST, Reuters, and the nested C dataset
- Support for semi-supervised and noisy scenarios
- Extensible to custom datasets

## Requirements

- Python 3.x
- TensorFlow 1.15
- Keras 2.3

Install dependencies:

```bash
pip install -r requirements.txt
```

On macOS, `wget` is also required to download the Reuters dataset:

```bash
brew install wget
```

## Quick Start

### Built-in datasets

```bash
cd src/applications
python run.py --gpu=0 --dset=mnist
```

Available datasets: `mnist` | `reuters` | `cc` | `cc_semisup`

### Reuters preprocessing

```bash
cd data/reuters
./get_data.sh
python make_reuters.py
```

### Custom dataset

Pass `(x_train, x_test, y_train, y_test)` to `get_data` in `src/core/data.py`, configure hyperparameters, then call `run_net()`. See the example below.

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

See `src/applications/run.py` for hyperparameter details.

## Project structure

```
spectral-clustering-demo/
├── data/reuters/          # Reuters dataset scripts
├── src/
│   ├── applications/      # Entry points and run scripts
│   ├── core/              # Networks, training, and data processing
│   └── pretrain_weights/  # Pretrained weight configs
├── requirements.txt
└── README.md
```
