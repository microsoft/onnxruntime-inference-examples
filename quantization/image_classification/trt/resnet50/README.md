# ONNX PTQ overview
Following is the end-to-end example using ORT quantization tool to quantize ONNX model, specifially image classification model, and run/evaluate the quantized model with TRT EP.  

## Environment setup
### dataset
First, prepare the dataset for calibration. TensorRT recommends calibration data size to be at least 500 for CNN and ViT models.
Generally, the dataset used for calibration should differ from the one used for evaluation. However, to simplify the sample code, we will use the same dataset for both calibration and evaluation. We recommend utilizing the ImageNet 2012 classification dataset for this purpose.

In addition to the sample code we provide below, TensorRT model optimizer which leverages torchvision.datasets already provides the ability to work with ImageNet dataset.

#### Prepare ImageNet dataset
You can either download from [Kaggle](https://www.kaggle.com/c/imagenet-object-localization-challenge/data) or origianl image-net website: val [tarball](https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar) and devkit [tarball](https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz)
```shell
mkdir ILSVRC2012
cd ILSVRC2012
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --no-check-certificate
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz --no-check-certificate
```
Untar the tarballs to `val` and `ILSVRC2012_devkit_t12` folder separately.

The dataset layout should look like below and the sample code expects this dataset layout

```
|-- ILSVRC2012_devkit_t12
|   |-- COPYING
|   |-- data
|   |   |-- ILSVRC2012_validation_ground_truth.txt
|   |   `-- meta.mat
|   |-- evaluation
|   |   |-- VOCreadrecxml.m
|   |   |-- VOCreadxml.m
|   |   |-- VOCxml2struct.m
|   |   |-- compute_overlap.m
|   |   |-- demo.val.pred.det.txt
|   |   |-- demo.val.pred.txt
|   |   |-- demo_eval.m
|   |   |-- eval_flat.m
|   |   |-- eval_localization_flat.m
|   |   |-- get_class2node.m
|   |   `-- make_hash.m
|   `-- readme.txt
|-- meta.bin
|-- synset_words.txt
`-- val
    |-- ILSVRC2012_val_00000001.JPEG
    |-- ILSVRC2012_val_00000002.JPEG
    |-- ILSVRC2012_val_00000003.JPEG
...
```

However, if you are using ImageNet, then please run following command to reconstruct the layout to be grouped by class.
```shell
cd val/
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
```
```
|-- ILSVRC2012_devkit_t12
|   |-- COPYING
|   |-- data
|   |   |-- ILSVRC2012_validation_ground_truth.txt
|   |   `-- meta.mat
|   |-- evaluation
|   |   |-- VOCreadrecxml.m
|   |   |-- VOCreadxml.m
|   |   |-- VOCxml2struct.m
|   |   |-- compute_overlap.m
|   |   |-- demo.val.pred.det.txt
|   |   |-- demo.val.pred.txt
|   |   |-- demo_eval.m
|   |   |-- eval_flat.m
|   |   |-- eval_localization_flat.m
|   |   |-- get_class2node.m
|   |   `-- make_hash.m
|   `-- readme.txt
|-- meta.bin
`-- val
    |-- n01440764
    |   |-- ILSVRC2012_val_00000293.JPEG
    |   |-- ILSVRC2012_val_00002138.JPEG
    |   |-- ILSVRC2012_val_00003014.JPEG
...
```
Lastly, download `synset_words.txt` from https://github.com/HoldenCaulfieldRye/caffe/blob/master/data/ilsvrc12/synset_words.txt into `ILSVRC2012` (top-level folder)

## Quantize an ONNX model

