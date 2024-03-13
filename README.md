# citiies_contrastive_learning

## TL;DR 

The below serves as a very rough and simple explanation for the project and the repository. For an elaborate presentation and discussion of the experimental performance of VICREG model based on varying remote sensing datasets used for backbone training, varying backbone architectures and augmentation regimes as well as comparative performances with respect to pretrained supervised backbones on well-established datasets, please refer to the file 'final_report.pdf' above.

## Preprocessing

The codes here are used to generate the csv file containing each image sample provided in the SN7 Dataset along with their labels which are possibly
the states, cities or countries each image belongs to extracted from their corresponding GeoJSON files. These csv files are used by the custom SN7 Dataset classes defined in the scripts. Also, each 1024x1024 image is though to have 16 256x256 patches (as can be seen form the generated csv files) which are then loaded as seperate images to the models. Also split the data indices into training and validation sets.

## Supervised Setting

The ~17000 training images are used to train a Resnet18 network with cross entropy loss for 100 epochs where countries are used as labels (31 distinct 
countries in total). The remaining ~5600 validation images are used to evaluate the performance of the trained model where the top1 accuracy was measured 
to be ~87%.

## Unsupervised-VicReg

The VicReg implementation trained and evaluated seperately on ImageNet and SN7 Datasets. Both codes support distributed training.

# ImageNet

The VicReg implementation with a Resnet50 backbone trained and evaluated on ImageNet Datasets. Use the following command lines to run:

1) Training

```
python -m torch.distributed.launch --nproc_per_node=3 main_vicreg_unchanged.py --data-dir '/path/to/imagenet/train' --exp-dir 'path/to/exp --arch resnet50 --epochs 100 --batch-size 512 --base-lr 0.3
```

2) Semi-Supervised Setting (Backbone Weights Finetuned on train-perc% of the training data):

```
python evaluate.py --pretrained 'path/to/model.pth' --batch-size 128 --data-dir  '/path/to/imagenet/train' --exp-dir path/to/exp --weights finetune --train-perc 1 --epochs 20 --lr-backbone 0.03 --lr-head 0.08 --weight-decay 0
```

3) Linear Evaluation (Backbone Weights Frozen with Linear Layer Trained on Whole Training Set):

```
python evaluate.py --pretrained 'path/to/model.pth' --batch-size 128 --data-dir  '/path/to/imagenet/train' --exp-dir path/to/exp --weights frozen --epochs 20 --lr-head 0.02
```

# SN7

The VicReg implementation with a Resne18 backbone trained and evaluated on SN7 Datasets. Use the following command lines to run:

1) Training: (At the end of 100 epochs: 16.88 Overall Loss, 0.003 MSE Loss, 0.56 standard-deviation loss, 2.72 covariance loss)

```
python -m torch.distributed.launch --nproc_per_node=2 main_vicreg_sn7.py --exp-dir exp/ --arch resnet18 --epochs 100 --batch-size 256 --base-lr 0.3
```

2) Semi-Supervised Setting (Backbone Weights Finetuned on train-perc% of the training data):

```
python evaluate_sn7.py --pretrained exp/trained_model.pth --batch-size 128 --exp-dir exp --weights finetune --train-perc 1 --epochs 20 --lr-backbone 0.03 --lr-head 0.08 --weight-decay 0 --arch resnet18
```

| - | 1% Train Percentage | 10% Train Percentage | Linear Layer on Frozen Features |
| ------------- | ------------- | ------------- | ------------- |
| Top1  | 0.47  | 0.73  | 0.826 |
| Top5  | 0.72 | 0.95  | 0.99 |

3) Linear Evaluation (Backbone Weights Frozen with Linear Layer Trained on Whole Training Set): (Top1: 0.83, Top5: 0.99)

```
python evaluate_sn7.py --pretrained exp/trained_model.pth --batch-size 128 --exp-dir exp --weights frozen --epochs 100 --lr-head 0.08  --arch resnet18
```
## Downstream Tasks

In this project, we mainly investigated any and all improvements a VICREG pretrained backbone had over the traditionally preferred backbones trained in a supervised manner, in particular for the case of downstream task performances in remote sensing data. For this purpose, we have used 3 tasks, each of which utilizes a large scale, non-synthetic remote sensing data. For a well-generalizable study, we have used three basic downstream tasks, namely, classification, detection and segmentation tasks, the code for each of which can be found under the folders 'eurosat_landcover_clsf_task', 'detectron2_crowdai_detection_task' and 'FloodNet_segmentation_task' respectively.





