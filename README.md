# Post-Training Neural Network Pruning using Graph Curvature

This repository contains the official implementation of **Post-Training Neural Network Pruning using Graph Curvature**, accepted to *Transactions on Machine Learning Research (TMLR), 2026*.

Our method studies neural network pruning from a graph-theoretic perspective and uses graph curvature to identify structurally important components of a trained neural network. The proposed approach enables one-shot post-training pruning without requiring retraining, providing an efficient way to reduce model complexity while preserving predictive performance.

## Paper

**Post-Training Neural Network Pruning using Graph Curvature**  
Shuhang Tan, Jayson Sia, Paul Bogdan, and Radoslav Ivanov  
*Transactions on Machine Learning Research (TMLR), 2026*

- [arXiv](https://arxiv.org/html/2601.16366v2)
- [OpenReview](https://openreview.net/forum?id=kwACVY73Ug)

## Citation

If you find this repository useful, please consider citing our paper:

```bibtex
@article{
tan2026posttraining,
title={Post-Training Neural Network Pruning using Graph Curvature},
author={Shuhang Tan and Jayson Sia and Paul Bogdan and Radoslav Ivanov},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2026},
url={https://openreview.net/forum?id=kwACVY73Ug},
note={}
}
```

## Neural Graph Definition (Default: `w4`)

- `--metric w4`

---

## Example Command

(See `run_vgg.sh` for a full example)

```bash
python removal.py --image 1 --metric w4 --model_type cnn --model_name ori \
--model_path CNN/models/cnn/ \
--img_res_path res/w4/relu/cifarori_vgg9_new_curv0/ \
--dataset cifar10 \
--community 1 \
--edge 0 \
--activation relu \
--sample_num 1 \
--img_data_path res/w4/relu/cifarori_vgg9/ \
--alpha 0.9
```

---

## Main Script

- `removal.py` – main pruning script

---

## Arguments

| Argument | Description |
|--------|-------------|
| `--image` | Image classifier mode |
| `--metric` | Neural graph metric (default: `w4`) |
| `--model_name` | Training method: `ori` (CE), `wd` (Weight Decay), `adv` (Adversarial Training) |
| `--model_path` | Path to pretrained model |
| `--img_res_path` | Path to save pruning results |
| `--dataset` | Dataset name (e.g., `cifar10`) |
| `--community` | `1`: calculate graph curvature |
| `--edge` | `1`: count results and start pruning |
| `--activation` | Activation function (`relu` or `tanh`) |
| `--sample_num` | Number of calibration samples per class used for curvature calculation |
| `--img_data_path` | Path to curvature results used for pruning |
| `--alpha` | Parameter for curvature calculation (Definition 5 in the paper) |


#### dataset: https://drive.google.com/drive/folders/1hmOSsB2mjwcBEO3eIjFsDB_9x1uJDjqb?usp=drive_link


