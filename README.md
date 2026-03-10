# Official code for Post Training NN Prunng using Graph Curvature

## Neural Graph Definitions (Default w4)

- --metric w4

## run commond (see run_vgg.sh as an example)

'python removal.py --image 1  --metric w4 --model_type cnn --model_name ori --model_path CNN/models/cnn/ --img_res_path res/w4/relu/cifarori_vgg9_new_curv0/ --dataset cifar10 --community 1 --edge 0 --activation relu --sample_num 1 --img_data_path res/w4/relu/cifarori_vgg9/ --alpha 0.9'

- removal.py (main file)

- --image (image classifier)

- --metric (default w4)

- --model_name (ori, wd, adv denote as CE, WD, AT)

- --img_res_path (results saved path)

- --community (1: calculate curvature)

- --edge (1: count results and start pruning)

- --activation (relu or tanh)

- --sample_num (number of calibration set used for curvature calculation, per label)

- --img_data_path (curvature results path used for pruning)

- --alpha (for curvature calculation, Def. 5 in the paper)
