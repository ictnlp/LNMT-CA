# LNMT-CA
This is a PyTorch implementation for EMNLP 2022 main conference paper [Low-resource Neural Machine Translation with Cross-modal Alignment](https://aclanthology.org/2022.emnlp-main.689/).

## Training a Model on De,Fr,Cs -> En

### Enviroment Configuration

1. Clone this repository:

```shell
git clone https://github.com/ictnlp/LNMT-CA
cd LNMT-CA/
```

2. Please make sure you have installed PyTorch, and then install fairseq and other packages as follows:

```shell
pip install --editable ./
python3 setup.py install --user
python3 setup.py build_ext --inplace
```

### Data Preparation

1. First make a directory to store the dataset:

```shell
SRC1=de
SRC2=fr
SRC3=cs

IMG_ROOT = data/img/
cd $IMG_ROOT

mkdir -p raw_images
mkdir $SRC1 $SRC2 $SRC3
```

2. Download the dataset

We have provided the image lists, raw data and BPE processed data. We combined three languages into one source text file, and name it as "mul-en" direction. In the source text, each language has an identifier, such as "[DE], [FR], [CS]". 

The Multi30K images can be downloaded [here](https://forms.illinois.edu/sec/229675), COCO images can be downloaded [here](https://cocodataset.org/#download)(Choose 2014 train, val and test images), and VizWiz images can be downloaed [here](https://vizwiz.org/tasks-and-datasets/image-captioning/)

```shell

cd $IMG_ROOT/raw_images

# Download the image data here

```

3. Extract the image feature

```shell
cd CLIP
python extract_features.py
```

4. Finally, the directory "data" should look like this:

```
.
├── text
│   ├── raw
│   │    ├──train.de
│   │    ├──train.fr
│   │    ├──......
│   ├── bpe
│   │    ├──train.mul
│   │    ├──train.en
│   │    ├──......
└── img
│   ├── raw_images
│   ├── de_img_name.txt
│   ├── fr_img_name.txt
│   ├── cs_img_name.txt
│   ├── de
│   │    ├── de_vit_clip_avg.npy
│   │    ├── de_vit_clip_0.npy
│   │    ├──......
│   ├── fr
│   ├── cs
```

### Data Preprocess

2. Use `fairseq-preprocess` command to convert the BPE texts into fairseq formats.

```shell
TEXT=data/text/bpe/mul-en
fairseq-preprocess --source-lang mul --target-lang en --trainpref ${TEXT}/train --validpref ${TEXT}/val  --testpref ${TEXT}/test_2016,${TEXT}/test_2017,${TEXT}/test_mscoco --destdir data-bin/multilingual-60k  --joined-dictionary --workers=20 
```


### Training
1. Train the model with sentence-level contrastive learning loss for 40-50 epochs:

```shell
exp=multilingual-60k
fairseq-train data-bin/multilingual-60k --task translation  --source-lang mul --target-lang en --arch transformer --dropout 0.3  --share-all-embeddings  \
    --image_root data/img/ --de_sen 60136 --fr_sen 60136 --cs_sen 60136 \
    --sen_tem 0.007 --token_tem 0.1 \
    --sentence_level True --token_level False \
    --encoder-layers 6 --decoder-layers 6 \
    --encoder-embed-dim 512 --decoder-embed-dim 512 \
    --encoder-ffn-embed-dim 1024 --decoder-ffn-embed-dim 1024 \
    --encoder-attention-heads 4 --decoder-attention-heads 4 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 2000 \
    --lr 0.005 \
    --criterion label_smoothed_cross_entropy_contrastive --label-smoothing 0.1 --weight-decay 0.0 \
    --max-tokens 4096 \
    --update-freq 4 --no-progress-bar --log-format json --log-interval 100 \
    --keep-last-epochs 10 \
    --save-dir data/checkpoints/$exp \
    --ddp-backend=no_c10d \
    --patience 10 \
    --left-pad-source False | tee experiment/logs/$exp.txt 
```
2. Train the model with sentence-level and token-level contrastive learning loss up to 60-70 epochs:

```shell
fairseq-train data-bin/multilingual-60k --task translation  --source-lang mul --target-lang en --arch transformer --dropout 0.3  --share-all-embeddings  \
    --image_root data/img/ --de_sen 60136 --fr_sen 60136 --cs_sen 60136 \
    --sen_tem 0.007 --token_tem 0.1 \
    --sentence_level True --token_level True \
    --encoder-layers 6 --decoder-layers 6 \
    --encoder-embed-dim 512 --decoder-embed-dim 512 \
    --encoder-ffn-embed-dim 1024 --decoder-ffn-embed-dim 1024 \
    --encoder-attention-heads 4 --decoder-attention-heads 4 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 2000 \
    --lr 0.005 \
    --criterion label_smoothed_cross_entropy_contrastive --label-smoothing 0.1 --weight-decay 0.0 \
    --max-tokens 4096 \
    --update-freq 4 --no-progress-bar --log-format json --log-interval 100 \
    --keep-last-epochs 10 \
    --save-dir experiment/checkpoints/$exp \
    --ddp-backend=no_c10d \
    --patience 10 \
    --left-pad-source False | tee experiment/logs/$exp.txt 
```

3. If you want to train the model with your own data, please remember to change the "de_sen", "fe_sen", "cs_sen", which means the number of sentence for each language


### Evaluate

1. Run the following script to average the last 5 checkpoints and evaluate on the three Multi30K test sets:

```shell
$MODEL=multilingual-60k
$DATASET=multilingual-60k

sh test_avg.sh $MODEL test $DATASET 5
sh test_avg.sh $MODEL test1 $DATASET 5
sh test_avg.sh $MODEL test2 $DATASET 5
```

The result will be stored at "LNMT-CA/result/"


## Citation
In this repository is useful for you, please cite as:

```
@inproceedings{yang-etal-2022-low,
    title = "Low-resource Neural Machine Translation with Cross-modal Alignment",
    author = "Yang, Zhe and Fang, Qingkai and Feng, Yang",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
}
```

## Contact

If you have any questions, feel free to contact me at `yangzhe22s1@ict.ac.cn`.
