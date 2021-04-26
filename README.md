
# Steps to run and replicate results
### Alex Su, Anish Saha, Chahat Chawla

## Follow these steps in the object_relation_transformer folder

0. conda activate srnlp

#### JSON stuff
1. Json is in data/
2. python scripts/prepro_labels.py --input_json data/dataset_wordseye.json --output_json data/wordseyetalk.json --output_h5 data/wordseyetalk
3. python scripts/prepro_ngrams.py --input_json data/dataset_wordseye.json --dict_json data/wordseyetalk.json --output_pkl data/wordseye-train --split train

#### Image Stuff
4. export IMAGE_ROOT=wordseye_images
5. mkdir $IMAGE_ROOT
6. pushd $IMAGE_ROOT
7. unzip created_data.zip
8. rm created_data.zip
9. popd
10. export PYTHONPATH=.
11. python scripts/prepro_feats.py --input_json data/dataset_wordseye.json --output_dir data/wordseyetalk --images_root $IMAGE_ROOT

#### Bottom-up feats
12. mkdir data/bu_data; cd data/bu_data
13. (From "https://github.com/airsplay/py-bottom-up-attention" (GPU ec2)) run "conda activate nlp_p3; python py-bottom-up-attention/demo/detectron2_mscoco_proposal_maxnms.py --split <...>;" to generate the TSVs
14. Place TSVs in data/bu_data/trainval/
15. python scripts/make_bu_data.py --output_dir data/wordseyebu

16. python scripts/prepro_bbox_relative_coords.py --input_json data/dataset_wordseye.json --input_box_dir data/wordseyebu_box --output_dir data/wordseyebu_box_relative --image_root $IMAGE_ROOT


#### Evaluation:
17. python eval.py --dump_images 0 --num_images 2 --model log_relation_transformer_bu/model-best.pth \
--infos_path log_relation_transformer_bu/infos_relation_transformer_bu-best.pkl --image_root $IMAGE_ROOT \
--input_json data/wordseyetalk.json --input_label_h5 data/wordseyetalk_label.h5  --input_fc_dir data/wordseyebu_fc --input_att_dir data/wordseyebu_att \
--input_box_dir data/wordseyebu_box --input_rel_box_dir data/wordseyebu_box_relative --language_eval 1
