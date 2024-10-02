# ONNX PTQ for using TensorRT EP
Following is the end-to-end example using ORT quantization tool to quantize ONNX model and run/evaluate the quantized model with TRT EP.  

## Environment setup
### dataset
We suggest to use ImageNet 2012 classification dataset to do the model calibration and evaluation. In addition to the sample code we provide below, TensorRT model optimizer which leverages torchvision.datasets already provides
the ability to work with ImageNet dataset.

#### Prepare ImageNet dataset
You can either download from [Kaggle](https://www.kaggle.com/c/imagenet-object-localization-challenge/data) or origianl image-net website: val [tarball](https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar) and devkit [tarball](https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz)
```shell
mkdir ILSVRC2012
cd ILSVRC2012
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --no-check-certificate
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz --no-check-certificate
```
Untar the tarballs to `val` and `ILSVRC2012_devkit_t12` folder separately.

The dataset layout should look like this:

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
Note: If the data in `val` folder is not grouped by class, please run following command to reconstruct the layout
```shell
cd val/
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
```
