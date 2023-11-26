  <h1>Turtle Conservation: Drones & AI in Action</h1>
<p>
    <img alt="Version" src="https://img.shields.io/badge/version-0.0.1-blue.svg?cacheSeconds=2592000" />
    <a href="https://apache.org/licenses/LICENSE-2.0.txt" target="_blank">
        <img alt="License: Apache License 2.0" src="https://img.shields.io/badge/License-Apache License 2.0-yellow.svg" />
    </a>
    <img alt="Python" src="https://img.shields.io/badge/python-v3.9-green" />
    <img alt="Conda" src="https://img.shields.io/badge/conda%7Cconda--forge-v3.7.1-important" />
</p>

<div style="text-align: justify">

> This project addresses the urgent challenge of conserving endangered turtle species by harnessing the capabilities of advanced drone technology and artificial intelligence. In response to the constraints of traditional conservation methods, our innovative approach integrates high-resolution drone imagery with a sophisticated AI model meticulously tuned for the accurate detection and monitoring of turtles in their natural habitats.
> This breakthrough streamlines the monitoring process and minimizes human intrusion into sensitive ecosystems. The significance of our work lies in its potential to revolutionize turtle conservation efforts, providing a scalable, efficient, and less invasive method to safeguard these crucial species. This contribution aligns with global biodiversity preservation goals, significantly striding towards sustainable and impactful wildlife conservation. 

</div>

## Folder structure

>The root folder contains different folders for each of the approaches we tried.

The project utilizes parser approach that allows usage of multiple flags that define hyperparameters and methodologies that could be applied in each run.

### base_model/

# cnn_sea_turtle_detection

[![DOI](https://zenodo.org/badge/158115622.svg)](https://zenodo.org/badge/latestdoi/158115622)

## Code for Methods in Ecology and Evolution paper: "A Convolutional Neural Network for Detecting Sea Turtles in Drone Imagery"

#### Paper can be accessed at: https://doi.org/10.1111/2041-210X.13132

### Using this code:

Running run.sh in bash will run the full turtle detection workflow (inside the base model folder).

#### Notes:
* Python 2.7 is required and nonstandard python packages necessary are: numpy, scipy, keras, tables, and hdf5storage
* This setup runs on preprocessed imagery contained in the .mat file. Full turtle image data along with labels for independent machine learning development can be found at doi:10.5061/dryad.5h06vv2

#### File Details:
* data.py                 
  * defines utility functions for model creation and matlab ingestion functions
* cnn_predict_stack.py
  * run prediction on processed images
* DukeTurtle_info.h5
  * Trained model weights file
* DukeTurtle_info.json
  * Model definition file
* DukeTurtle_test.mat
  * processed and tiled RGB image data that is fed into the model. Training / validation split is 85% train / 15% validation 

### data_imbalance/main.py

<div style="text-align: justify">
    This part contains Python3 version of the code provided in the original paper that was also rewritten using Pytorch. Here it is possible to try multiple techniques that aim to tackle data-imbalance issue as well as modify hyperparameters.
</div>

| Parameter            | Default  | Description                                    |
|:---------------------|:---------|:-----------------------------------------------|
| `--lr` | `0.025`     | Learning rate          |
| `--experiment_name` | `exp`     | Name of the experiment |
| `--output_folder`  | `results`    | Output folder path                                  |
| `--epochs`  | `10`    | Number of epochs                                 |
| `--batch_size` | `64` | Batch size                |
| `--label_smoothing`    | `0.2`   | Specifies the amount of smoothing when computing the loss, where 0.0 means no smoothing. The targets become a mixture of the original ground truth and a uniform distribution |
| `--l2_lambda`  | `0.01`  | Value for L2-regularization computes the average of the squared differences between actual values and predicted values                                 |
| `--enable_aug_rhf` | `False` | Enable Random Horizontal flip as a transpormation for input image                |
| `--enable_aug_rvf` | `False` | Enable Random Vertical flip as a transpormation for input image                |
| `--enable_aug_rr` | `False` | Enable Random Rotation flip as a transpormation for input image                |
| `--enable_ins_weights` | `False` | Enable Inverse of Number of Samples as a weight equation for loss                 |
| `--enable_root_weights` | `False` | Enable Inverse of Square Root of Number of Samples  as a weight equation for loss                          |
| `--enable_label_smoothing` | `False` | Enable lable smoothing parameter usage in loss function                |


```sh
  cd data_imbalance
  # Run with default values
  python main.py
  # Defining parameters (these are recomended)
  python main.py --enable_aug_rhf True --enable_aug_rvf True --enable_aug_rr True --epochs 5 --batch_size 128
```

### data_imbalance/view_csv.ipynb

<div style="text-align: justify">
    This notebook provides description regarding the steps taken for data analysis. Analyzing imbalanced datasets is crucial in machine learning for several reasons. Imbalanced datasets, where the distribution of classes is uneven, can lead to biased models that favor the majority class and perform poorly on minority classes. It showcases steps taken to gain better understanding of data as well as generation of new annotation files.
</div>

### small_object_detection/main.py

<div style="text-align: justify">
    This part contains Python3 version of the code provided in the original paper that was also rewritten using Pytorch. Here it is possible to try multiple techniques that aim to tackle data-imbalance issue as well as modify hyperparameters.
</div>

| Parameter            | Default  | Description                                    |
|:---------------------|:---------|:-----------------------------------------------|
| `--lr` | `0.025`     | Learning rate          |
| `--experiment_name` | `exp`     | Name of the experiment |
| `--output_folder`  | `results`    | Output folder path                                  |
| `--epochs`  | `10`    | Number of epochs                                 |
| `--batch_size` | `64` | Batch size                |
| `--label_smoothing`    | `0.2`   | Specifies the amount of smoothing when computing the loss, where 0.0 means no smoothing. The targets become a mixture of the original ground truth and a uniform distribution |
| `--l2_lambda`  | `0.01`  | Value for L2-regularization computes the average of the squared differences between actual values and predicted values                                 |


```sh
  cd small_object_detection
  # Run with default values
  python main.py
  # Defining parameters
  python main.py --epochs 50 --batch_size 128 --lr 0.234
```

### /SAM/SAM_Teast_1.ipynb

Segment Anything Model (SAM): a new AI model from Meta AI that can "cut out" any object, in any image, with a single click. SAM is a promptable segmentation system with zero-shot generalization to unfamiliar objects and images, without the need for additional training. This notebook is is built on the [official notebook](https://colab.research.google.com/github/facebookresearch/segment-anything/blob/main/notebooks/automatic_mask_generator_example.ipynb) prepared by Meta AI.

## How to download the dataset

Dataset is available at https://datadryad.org/stash/dataset/doi:10.5061/dryad.5h06vv2. You should download it and extract it the root folder.


## How to configure the environment

### With PIP (recommended)

<div style="text-align: justify">
    We provide a file <b>requirements.txt</b> to install all the dependencies
    with a package manager like <b><a target="_blank" href="https://pip.pypa.io/en/stable/cli/pip_install/">pip</a></b>.
</div>

```sh
# Create manually conda environment
conda create -n ai_701 python=3.9

# Install with pip
pip install -r requirements.txt
```

<div style="text-align: justify">
    2. When the environment completes its configuration, just access the environment
    and launch one of the configurations mentioned in the previous section:
</div>

```sh
cd data_imbalance

# Activate environment
conda activate ai_701

# e.g. Run with epochs 5 and batch_size 128
python main.py --enable_aug_rhf True --enable_aug_rvf True --enable_aug_rr True --epochs 5 --batch_size 128
```

### With Conda 

<div style="text-align: justify">
    1. Start by creating a new Python environment using the provided
    configuration file <b>environment.yml</b>:
</div>

```sh
conda env create -f environment.yml
```

<div style="text-align: justify">
    When completed, just execute the same step 2 as from the <b>With other environment managers</b> subsection.
</div>

## Run the current best model

<div style="text-align: justify">
    In the folder <b>best_model</b> you can find the model we were able to train.
</div>

#### Data Imbalance

```sh
cd data_imbalance
python main.py --is_test True
```

## Common issues

#### When running I've been asked for a Wandb (W&B) account

<div style="text-align: justify">
    We use <a target="_blank" href="https://wandb.ai/">wandb</a> to plot our results, you can create an account to
    visualize the progress of an execution, or just chose the option 3, it
    will ignore any logging.
</div>

```sh
wandb: (1) Create a W&B account
wandb: (2) Use an existing W&B account
wandb: (3) Don't visualize my results
wandb: Enter your choice: 
```

## Authors

|              |                       **Alvaro Cabrera**                        |                       **Yevheniia Kryklyvets**                       |
|--------------|:----------------------------------------------------------------------:|:------------------------------------------------------------------:|
| **Github**   |              [@alvaro-cabrera](https://github.com/alvaro-cabrera)              |          [@ETZ2017](https://github.com/ETZ2017)                 |

## Acknowledgements

Thanks to the creators of the open source code that was the seed of this project.

 - [patrickcgray](https://github.com/patrickcgray/cnn_sea_turtle_detection)

## License

Copyright © 2023 [Alvaro Cabrera, Yevheniia Kryklyvets].<br />
This project is [Apache License 2.0](https://apache.org/licenses/LICENSE-2.0.txt) licensed.