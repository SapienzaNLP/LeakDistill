# LeakDistill
This is the repo for [Incorporating Graph Information in Transformer-based AMR Parsing](https://arxiv.org/abs/2306.13467). The paper introduces a novel way to incorporate structural information at training time using Structural Adapters. This repo is an extension of SPRING [repo](https://github.com/SapienzaNLP/spring).
If you use our code, please reference this work in your paper:

```
@inproceedings{vasylenko-etal-2023-leakdistill,
    title = {Incorporating Graph Information in Transformer-based AMR Parsing},
    author = {Vasylenko, Pavlo and Huguet Cabot, Pere-Lluís and Martínez Lorenzo, Abelardo Carlos and Navigli, Roberto},
    booktitle = {Findings of ACL},
    year = {2023}
}
```
## Installation
```shell script
cd LeakDistill
pip install -r requirements.txt
pip install -e .
```

## Training
### LeakDistill

```shell script
python bin/train.py --config configs/config_leak_distill.yaml
```

### Graph Leakage Model

```shell script
python bin/train.py --config configs/config_leak.yaml
```

### Vanilla Knowledge Distillation

```shell script
python bin/train_kd.py --config configs/config_kd.yaml --teacher <path_to_checkpoint>
```

### SPRING 

```shell script
python bin/train.py --config configs/config_spring.yaml
```
## Pretrained Checkpoints

For any questions or inquiries, please contact Pavlo Vasylenko at vasylen.pavlo@gmail.com or Pere-Lluís Huguet Cabot at huguetcabot@babelscape.com

## Evaluation

```shell script
python bin/predict_amrs.py \
    --config configs/config_leak_distill.yaml \
    --datasets '<path_to_datasets>' \
    --gold-path data/tmp/amr2.0/gold.amr.txt \
    --pred-path data/tmp/amr2.0/pred.amr.txt \
    --beamsize 10 \
    --checkpoint <path_to_checkpoint> \
    --device cuda
```
if `datasets` is not specified the path is taken from the `config.test`.

`gold.amr.txt` and `pred.amr.txt` will contain, respectively, the concatenated gold and the predictions.

To reproduce our paper's results, you will also need need to run [BLINK](https://github.com/facebookresearch/BLINK) entity linking system on the prediction file (`data/tmp/amr2.0/pred.amr.txt` in the previous code snippet). 
To do so, you will need to install BLINK, and download their models:
```shell script
git clone https://github.com/facebookresearch/BLINK.git
cd BLINK
pip install -r requirements.txt
sh download_blink_models.sh
cd models
wget http://dl.fbaipublicfiles.com/BLINK//faiss_flat_index.pkl
cd ../..
```
Then, you will be able to launch the `blinkify.py` script:
```shell
python bin/blinkify.py \
    --datasets data/tmp/amr2.0/pred.amr.txt \
    --out data/tmp/amr2.0/pred.amr.blinkified.txt \
    --device cuda \
    --blink-models-dir BLINK/models
```
To have comparable Smatch scores you will also need to use the scripts available at https://github.com/mdtux89/amr-evaluation, which provide
results that are around ~0.3 Smatch points lower than those returned by `bin/predict_amrs.py`.

## License

This project is released under the CC-BY-NC-SA 4.0 license (see `LICENSE`). If you use LeakDistill, please reference the paper and put a link to this repo.
