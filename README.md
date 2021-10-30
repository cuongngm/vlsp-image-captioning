# Solution for VLSP image captioning task
### Install
```sh
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
# download pretrain and some necessary package from
download_weights.sh
```
### Prepare dataset
```sh
mkdir dataset
cd dataset
# download training data and unzip them in 'dataset' folder
gdown --id 1lbOTlksNA5a97_Ydqh84TE6Dm85Rsy60
# download private data and unzip them in 'dataset' folder
gdown --id 1rCDniCZNgaJ7WQUzPpzEwuXW5_WNyave
```
### Training
```sh
python train.py --decoder_mode lstm --batch_size 16 --checkpoint checkpoint/pretrain_coco.pth.tar --fine_tune_encoder True
```

### Inference
```sh
python caption.py --decoder_mode lstm --checkpoint checkpoint/model_best.pth.tar
```

### My solution result
| Method                   | Avg BLEU score (public test) | Avg BLEU score (private test)
|:--------------------------:|:-------:|:--------:|
| Resnet + LSTM (with pretrained COCO dataset) | 0.279 | 0.273 |
| Resnet + LSTM (without COCO dataset) | 0.263 |
| Resnet + Transformer | 0.265 |  |
| EfficientNet B7 + LSTM | 0.261 |  |
### Reference
I got a lot of code from [Image-Caption](https://github.com/RoyalSkye/Image-Caption), thanks to [@Jianan](https://github.com/RoyalSkye) 