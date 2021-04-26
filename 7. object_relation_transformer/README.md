# Object Relation Transformer

This is a PyTorch implementation of the Object Relation Transformer published in NeurIPS 2019. You can find the paper [here](https://papers.nips.cc/paper/9293-image-captioning-transforming-objects-into-words.pdf). This repository is largely based on code from Ruotian Luo's Self-critical Sequence Training for Image Captioning GitHub repo, which can be found [here](https://github.com/ruotianluo/self-critical.pytorch).

The primary additions are as follows:
* Relation transformer model
* Script to create reports for runs on MSCOCO


## Requirements
* Python 2.7 (because there is no [coco-caption](https://github.com/tylin/coco-caption) version for Python 3)
* PyTorch 0.4+ (along with torchvision)
* h5py
* scikit-image
* typing
* pyemd
* gensim
* [cider](https://github.com/ruotianluo/cider.git) (already added as a submodule). See `.gitmodules` and clone the referenced repo into
  the `object_relation_transformer` folder.  
* The [coco-caption](https://github.com/tylin/coco-caption) library,
  which is used for generating different evaluation metrics. To set it
  up, clone the repo into the `object_relation_transformer`
  folder. Make sure to keep the cloned repo folder name as
  `coco-caption` and also to run the `get_stanford_models.sh`
  script from within that repo.



## Data Preparation

### Download ResNet101 weights for feature extraction

Download the file `resnet101.pth` from [here](https://drive.google.com/drive/folders/0B7fNdx_jAqhtbVYzOURMdDNHSGM). Copy the weights to a folder `imagenet_weights` within the data folder:

```
mkdir data/imagenet_weights
cp /path/to/downloaded/weights/resnet101.pth data/imagenet_weights
```

### Download and preprocess the COCO captions

Download the [preprocessed COCO captions](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip) from Karpathy's homepage. Extract `dataset_coco.json` from the zip file and copy it in to `data/`. This file provides preprocessed captions and also standard train-val-test splits.

Then run:

```
$ python scripts/prepro_labels.py --input_json data/dataset_coco.json --output_json data/cocotalk.json --output_h5 data/cocotalk
```
`prepro_labels.py` will map all words that occur <= 5 times to a special `UNK` token, and create a vocabulary for all the remaining words. The image information and vocabulary are dumped into `data/cocotalk.json` and discretized caption data are dumped into `data/cocotalk_label.h5`.

Next run:
```
$ python scripts/prepro_ngrams.py --input_json data/dataset_coco.json --dict_json data/cocotalk.json --output_pkl data/coco-train --split train
```

This will preprocess the dataset and get the cache for calculating cider score.


### Download the COCO dataset and pre-extract the image features

Download the [COCO images](http://mscoco.org/dataset/#download) from the MSCOCO website.
We need 2014 training images and 2014 validation images. You should put the `train2014/` and `val2014/` folders in the same directory, denoted as `$IMAGE_ROOT`:

```
mkdir $IMAGE_ROOT
pushd $IMAGE_ROOT
wget http://images.cocodataset.org/zips/train2014.zip
unzip train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip
popd
wget https://msvocds.blob.core.windows.net/images/262993_z.jpg
mv 262993_z.jpg $IMAGE_ROOT/train2014/COCO_train2014_000000167126.jpg
```

The last two commands are needed to address an issue with a corrupted image in the MSCOCO dataset (see [here](https://github.com/karpathy/neuraltalk2/issues/4)). The prepro script will fail otherwise.


Then run:

```
$ python scripts/prepro_feats.py --input_json data/dataset_coco.json --output_dir data/cocotalk --images_root $IMAGE_ROOT
```

`prepro_feats.py` extracts the ResNet101 features (both fc feature and last conv feature) of each image. The features are saved in `data/cocotalk_fc` and `data/cocotalk_att`, and resulting files are about 200GB. Running this script may take a day or more, depending on hardware.

(Check the prepro scripts for more options, like other ResNet models or other attention sizes.)

### Download the Bottom-up features

Download the pre-extracted features from [here](https://github.com/peteanderson80/bottom-up-attention). For the paper, the adaptive features were used.

Do the following:
```
mkdir data/bu_data; cd data/bu_data
wget https://imagecaption.blob.core.windows.net/imagecaption/trainval.zip
unzip trainval.zip

```
The .zip file is around 22 GB.
Then return to the base directory and run:
```
python scripts/make_bu_data.py --output_dir data/cocobu
```

This will create `data/cocobu_fc`, `data/cocobu_att` and `data/cocobu_box`.


### Generate the relative bounding box coordinates for the Relation Transformer

Run the following:
```
python scripts/prepro_bbox_relative_coords.py --input_json data/dataset_coco.json --input_box_dir data/cocobu_box --output_dir data/cocobu_box_relative --image_root $IMAGE_ROOT
```
This should take a couple hours or so, depending on hardware.


## Model Training and Evaluation

### Standard cross-entropy loss training

```
python train.py --id relation_transformer_bu --caption_model relation_transformer --input_json data/cocotalk.json --input_fc_dir data/cocobu_fc --input_att_dir data/cocobu_att --input_box_dir data/cocobu_box --input_rel_box_dir data/cocobu_box_relative --input_label_h5 data/cocotalk_label.h5 --checkpoint_path log_relation_transformer_bu --noamopt --noamopt_warmup 10000 --label_smoothing 0.0 --batch_size 15 --learning_rate 5e-4 --num_layers 6 --input_encoding_size 512 --rnn_size 2048 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --save_checkpoint_every 6000 --language_eval 1 --val_images_use 5000 --max_epochs 30 --use_box 1
```

The train script will dump checkpoints into the folder specified by `--checkpoint_path` (default = `save/`). We only save the best-performing checkpoint on validation and the latest checkpoint to save disk space.

To resume training, you can specify `--start_from` option to be the path saving `infos.pkl` and `model.pth` (usually you could just set `--start_from` and `--checkpoint_path` to be the same).

If you have tensorflow, the loss histories are automatically dumped into `--checkpoint_path`, and can be visualized using tensorboard.

The current command uses scheduled sampling. You can also set scheduled_sampling_start to -1 to disable it.

If you'd like to evaluate BLEU/METEOR/CIDEr scores during training in addition to validation cross entropy loss, use `--language_eval 1` option, but don't forget to download the [coco-caption code](https://github.com/tylin/coco-caption) into `coco-caption` directory.

For more options, see `opts.py`.


The above training script should achieve a CIDEr-D score of about 115.


### Self-critical RL training

After training using cross-entropy loss, additional self-critical training produces signficant gains in CIDEr-D score.


First, copy the model from the pretrained model using cross entropy. (It's not mandatory to copy the model, just for back-up)
```
$ bash scripts/copy_model.sh relation_transformer_bu relation_transformer_bu_rl
```

Then:

```
python train.py --id relation_transformer_bu_rl --caption_model relation_transformer --input_json data/cocotalk.json --input_fc_dir data/cocobu_fc --input_att_dir data/cocobu_att --input_label_h5 data/cocotalk_label.h5  --input_box_dir data/cocobu_box --input_rel_box_dir data/cocobu_box_relative --input_label_h5 data/cocotalk_label.h5 --checkpoint_path log_relation_transformer_bu_rl --label_smoothing 0.0 --batch_size 10 --learning_rate 5e-4 --num_layers 6 --input_encoding_size 512 --rnn_size 2048 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --start_from log_transformer_bu_rl --save_checkpoint_every 6000 --language_eval 1 --val_images_use 5000 --self_critical_after 30 --max_epochs 60 --use_box 1
```

The above training script should achieve a CIDEr-D score of about 128.

### Evaluate on Karpathy's test split
To evaluate the cross-entropy model, run:

```
python eval.py --dump_images 0 --num_images 5000 --model log_relation_transformer_bu/model.pth --infos_path log_relation_transformer_bu/infos_relation_transformer_bu-best.pkl --image_root $IMAGE_ROOT --input_json data/cocotalk.json --input_label_h5 data/cocotalk_label.h5  --input_fc_dir data/cocobu_fc --input_att_dir data/cocobu_att --input_box_dir data/cocobu_box --input_rel_box_dir data/cocobu_box_relative --use_box 1 --language_eval 1
```

and for cross-entropy+RL run:

```
python eval.py --dump_images 0 --num_images 5000 --model log_relation_transformer_bu_rl/model.pth --infos_path log_relation_transformer_bu_rl/infos_relation_transformer_bu-best.pkl --image_root $IMAGE_ROOT --input_json data/cocotalk.json --input_label_h5 data/cocotalk_label.h5  --input_fc_dir data/cocobu_fc --input_att_dir data/cocobu_att --input_box_dir data/cocobu_box --input_rel_box_dir data/cocobu_box_relative --language_eval 1
```

## Visualization

### Visualize caption predictions
Place all your images of interest into a folder, e.g. `images`, and run
the eval script:

```
$ python eval.py --dump_images 1 --num_images 10 --model log_relation_transformer_bu/model.pth --infos_path log_relation_transformer_bu/infos_relation_transformer_bu-best.pkl --image_root $IMAGE_ROOT --input_json data/cocotalk.json --input_label_h5 data/cocotalk_label.h5  --input_fc_dir data/cocobu_fc --input_att_dir data/cocobu_att --input_box_dir data/cocobu_box --input_rel_box_dir data/cocobu_box_relative
```

This tells the `eval` script to run up to 10 images from the given folder. If you have a big GPU you can speed up the evaluation by increasing `batch_size`. Use `--num_images -1` to process all images. The eval script will create an `vis.json` file inside the `vis` folder, which can then be visualized with the provided HTML interface:

```
$ cd vis
$ python -m SimpleHTTPServer
```

Now visit `localhost:8000` in your browser and you should see your predicted captions.

### Generate reports from runs on MSCOCO

The [create_report.py](create_report.py) script can be used in order to generate HTML reports containing results from different runs. Please see the script for specific usage examples.

The script takes as input one or more pickle files containing results from runs on the MSCOCO dataset. It reads in the pickle files and creates a set of HTML files with tables and graphs generated from the different captioning evaluation metrics, as well as the generated image captions and corresponding metrics for individual images.

If more than one pickle file with results is provided as input, the script will also generate a report containing a comparison between the metrics generated by each pair of methods.


## Model Zoo and Results

The table below presents links to our pre-trained models, as well as results from our paper on the Karpathy test
split. Similar results should be obtained by running the respective commands in
[neurips_training_runs.sh](neurips_training_runs.sh). As learning rate scheduling was not fully optimized, these
values should only serve as a reference/expectation rather than what can be achieved with additional tuning.

The models are Copyright Verizon Media, licensed under the terms of the CC-BY-4.0 license. See associated
[license file](LICENSE-CC-BY-4.md).

Algorithm | CIDEr-D |SPICE | BLEU-1 | BLEU-4 | METEOR | ROUGE-L
:-- | :--: | :--: | :--: | :--: | :--: | :--:
[Up-Down + LSTM](http://wpc.D89ED.chicdn.net/00D89ED/object_relation_transformer/models/log_topdown_bu.zip) * | 106.6 | 19.9 | 75.6 | 32.9 | 26.5 | 55.4
[Up-Down + Transformer](http://wpc.D89ED.chicdn.net/00D89ED/object_relation_transformer/models/log_transformer_bu.zip) | 111.0 | 20.9 | 75.0 | 32.8 | 27.5 | 55.6
[Up-Down + Object Relation Transformer](http://wpc.D89ED.chicdn.net/00D89ED/object_relation_transformer/models/log_relation_transformer_bu.zip) | 112.6 | 20.8 | 75.6 |33.5 |27.6 | 56.0
[Up-Down + Object Relation Transformer](http://wpc.D89ED.chicdn.net/00D89ED/object_relation_transformer/models/log_relation_transformer_bu.zip) + Beamsize 2 | 115.4 | 21.2 | 76.6 | 35.5 | 28.0 | 56.6
[Up-Down + Object Relation Transformer + Self-Critical](http://wpc.D89ED.chicdn.net/00D89ED/object_relation_transformer/models/log_relation_transformer_bu_rl.zip) + Beamsize 5 | 128.3 | 22.6 | 80.5 | 38.6 | 28.7 | 58.4

\* Note that the pre-trained Up-Down + LSTM model above produces slightly better results than
reported, as it came from a different training run. We kept the older LSTM results in the table above for consistency
with our paper.

### Comparative Analysis

In addition, in the paper we also present a head-to-head comparison of the Object Relation Transformer against the "Up-Down + Transformer" model. (Results from the latter model are also included in the table above).
In the paper, we refer to this latter model as "Baseline Transformer", as it does not make use of geometry in its attention definition. The idea of the head-to-head comparison is to better understand the improvement
obtained by adding geometric attention to the Transformer, both quantitatively and qualitatively. The comparison consists of a set of evaluation metrics computed for each model on a per-image basis, as well as aggregated over all images.
It includes the results of paired t-tests, which test for statistically significant differences between the evaluation metrics resulting from each of the models. This comparison can be generated by running the commands in
[neurips_report_comands.sh](neurips_report_commands.sh). The commands first run the two aforementioned models on the MSCOCO test set and then generate the corresponding report containing the complete comparative analysis.


## Citation

If you find this repo useful, please consider citing (no obligation at all):

```
@article{herdade2019image,
  title={Image Captioning: Transforming Objects into Words},
  author={Herdade, Simao and Kappeler, Armin and Boakye, Kofi and Soares, Joao},
  journal={arXiv preprint arXiv:1906.05963},
  year={2019}
}
```

Of course, please cite the original paper of models you are using (you can find references in the model files).

## Contribute

Please refer to [the contributing.md file](Contributing.md) for information about how to get involved. We welcome
issues, questions, and pull requests.

Please be aware that we (the maintainers) are currently busy with other projects, so it make take some days before we
are able to get back to you. We do not foresee big changes to this repository going forward.

## Maintainers

Kofi Boakye: kaboakye@verizonmedia.com

Simao Herdade: sherdade@verizonmedia.com

Joao Soares: jvbsoares@verizonmedia.com

## License

This project is licensed under the terms of the MIT open source license. Please refer to [LICENSE](LICENSE) for the full terms.


## Acknowledgments

Thanks to [Ruotian Luo](https://github.com/ruotianluo) for the original code.
