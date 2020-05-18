Slide Classifier
================

Overview
--------

This model classifies a video frame from a lecture video according to the following 9 categories:

* audience
* audience_presenter
* audience_presenter_slide
* demo
* presenter
* presenter_slide
* presenter_whiteboard
* slide
* whiteboard

Pre-trained Models
------------------

Not completed yet.

Raw Data Download
-----------------

Not completed yet.

Script Descriptions
-------------------

* **class_cluster_scikit.py**: Implements ``KMeans`` and ``AffinityPropagation`` from ``sklearn.cluster`` to provde a ``Cluster`` class. The code is documented in file. The purpose is to add feature vectors using ``add()``, then cluster the features, and finally return a list of files and their corresponding cluster centroids with ``create_move_list()``. Three important functions and their use cases follow:

    * ``create_move_list()`` is called by ``End-To-End/cluster.py`` and returns a list of filenames and their corresponding clusters.
    * ``calculate_best_k()`` generates a graph (saved to ``best_k_value.png`` if using Agg matplotlib backend) that graphs the cost (squared error) as a function of the number of centroids (value of k) if the algorithm is ``"kmeans"``. The point at which the graph becomes essentially linear is the optimal value of k.
    * ``visualize()`` creates a tensorboard projection of the cluster for simplified viewing and understanding.

* **class_cluster_faiss.py**: An outdated version of **class_cluster_scikit** that uses `facebookresearch/faiss <https://github.com/facebookresearch/faiss>`_ (specifically the kmeans implementation `documented here <https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks:-clustering,-PCA,-quantization>`_) to provide a ``Cluster`` class. More details in the ``class_cluster_scikit`` entry above.
* **custom_nnmodules.py**: Provides a few custom (copied from `fastai <https://github.com/fastai/fastai>`_) nn.Modules.
* **inference.py**: Sets up model and provides ``get_prediction()``, which takes an image and returns a prediction and extracted features. 
* **lr_finder.py**: Slightly modified (allows usage of matplotlib Agg backend) code from `davidtvs/pytorch-lr-finder <https://github.com/davidtvs/pytorch-lr-finder>`_ to find the best learning rate.
* **mish.py**: Code for the mish activation function.
* **slide-classifier-fastai.ipynb**: Notebook to train simple fastai classifier on the dataset in ``Dataset/classifier-data``. It is outdated and not supported and only remains in the repository as an example.
* **slide-classifier-pytorch.py**: The main model code which uses advanced features such as the AdamW optimizer and a modified ResNet that allows for more effective pre-training/feature extracting.

Slide-Classifier-Pytorch Help
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Output of ``python slide-classifier-pytorch.py --help``:

.. code-block:: bash

    usage: slide-classifier-pytorch.py [-h] [-a ARCH] [-j N] [--epochs N]
                                    [--start-epoch N] [-b N] [--lr LR]
                                    [--momentum M] [--wd W] [-k K] [--alpha N]
                                    [--eps N] [-p N] [--resume PATH] [-e]
                                    [--pretrained] [--seed SEED] [--gpu GPU]
                                    [--random_split] [--relu_to_mish]
                                    [--feature_extract {normal,advanced}]
                                    [--find_lr] [-o OPTIM]
                                    [--tensorboard-model] [--tensorboard PATH]
                                    [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                                    DIR

    PyTorch Slide Classifier Training

    positional arguments:
    DIR                   path to dataset

    optional arguments:
    -h, --help            show this help message and exit
    -a ARCH, --arch ARCH  model architecture: alexnet | densenet121 |
                            densenet161 | densenet169 | densenet201 | googlenet |
                            inception_v3 | mnasnet0_5 | mnasnet0_75 | mnasnet1_0 |
                            mnasnet1_3 | mobilenet_v2 | resnet101 | resnet152 |
                            resnet18 | resnet34 | resnet50 | resnext101_32x8d |
                            resnext50_32x4d | shufflenet_v2_x0_5 |
                            shufflenet_v2_x1_0 | shufflenet_v2_x1_5 |
                            shufflenet_v2_x2_0 | squeezenet1_0 | squeezenet1_1 |
                            vgg11 | vgg11_bn | vgg13 | vgg13_bn | vgg16 | vgg16_bn
                            | vgg19 | vgg19_bn | wide_resnet101_2 |
                            wide_resnet50_2 | efficientnet-b0 | efficientnet-b1 |
                            efficientnet-b2 | efficientnet-b3 | efficientnet-b4 |
                            efficientnet-b5 | efficientnet-b6 (default: resnet34)
    -j N, --workers N     number of data loading workers (default: 4)
    --epochs N            number of total epochs to run (default: 6)
    --start-epoch N       manual epoch number (useful on restarts)
    -b N, --batch-size N  mini-batch size (default: 16)
    --lr LR, --learning-rate LR
                            initial learning rate
    --momentum M          momentum. Ranger optimizer suggests 0.95.
    --wd W, --weight-decay W
                            weight decay (default: 1e-2)
    -k K, --ranger-k K    Ranger (LookAhead) optimizer k value (default: 6)
    --alpha N             Optimizer alpha parameter (default: 0.999)
    --eps N               Optimizer eps parameter (default: 1e-8)
    -p N, --print-freq N  print frequency (default: -1)
    --resume PATH         path to latest checkpoint (default: none)
    -e, --evaluate        evaluate model on validation set and generate overall
                            statistics/confusion matrix
    --pretrained          use pre-trained model
    --seed SEED           seed for initializing training.
    --gpu GPU             GPU id to use.
    --random_split        use random_split to create train and val set instead
                            of train and val folders
    --relu_to_mish        convert any relu activations to mish activations
    --feature_extract {normal,advanced}
                            If False, we finetune the whole model. When normal we
                            only update the reshaped layer params. When advanced
                            use fastai version of feature extracting (add fancy
                            group of layers and only update this group and
                            BatchNorm)
    --find_lr             Flag for lr_finder.
    -o OPTIM, --optimizer OPTIM
                            Optimizer to use (default=AdamW)
    --tensorboard-model   Flag to write the model to tensorboard. Action is RAM
                            intensive.
    --tensorboard PATH    Path to tensorboard logdir. Tensorboard not used if
                            not set.
    -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --log {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                            Set the logging level (default: 'Info').

Experiments
-----------

Training Configurations (Commands)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **efficientnet-ranger:**

    .. code-block:: bash

        python slide-classifier-pytorch.py -a efficientnet-b0 --random_split --pretrained --feature_extract advanced  ../../Dataset/classifier-data -b 10 --epochs 10 --tensorboard runs/efficientnet-adamw

2. **efficientnet-adamw:**

    .. code-block:: bash
        
        python slide-classifier-pytorch.py -a efficientnet-b0 --random_split --pretrained --feature_extract advanced  ../../Dataset/classifier-data -b 10 --epochs 10 --tensorboard runs/efficientnet-adamw-optimized --momentum 0.95 --eps 1e-5 --wd 0

3. **resnet34-adamw:**

    .. code-block:: bash
        
        python slide-classifier-pytorch.py -a resnet34 --random_split --pretrained --feature_extract advanced  ../../Dataset/classifier-data -b 10 --epochs 10 --tensorboard runs/resnet34-adamw

4. **resnet34-ranger:**

    .. code-block:: bash

        python slide-classifier-pytorch.py -a resnet34 --random_split --pretrained --feature_extract advanced  ../../Dataset/classifier-data -b 10 --epochs 10 --tensorboard runs/resnet34-ranger -o ranger -k 3 --momentum 0.95 --eps 1e-5 --wd 0

5. **resnet34-adamw-mish:**

    .. code-block:: bash

        python slide-classifier-pytorch.py -a resnet34 --random_split --pretrained --feature_extract advanced  ../../Dataset/classifier-data -b 10 --epochs 10 --tensorboard runs/resnet34-adamw-mish --relu_to_mish

Results
^^^^^^^

+--------------------------+-----------+--------+----------+---------+
| Class                    | Precision | Recall | F1-Score | Support |
+==========================+===========+========+==========+=========+
| audience                 |           |        |          |         |
+--------------------------+-----------+--------+----------+---------+
| audience_presenter       |           |        |          |         |
+--------------------------+-----------+--------+----------+---------+
| audience_presenter_slide |           |        |          |         |
+--------------------------+-----------+--------+----------+---------+
| demo                     |           |        |          |         |
+--------------------------+-----------+--------+----------+---------+
| presenter                |           |        |          |         |
+--------------------------+-----------+--------+----------+---------+
| presenter_slide          |           |        |          |         |
+--------------------------+-----------+--------+----------+---------+
| presenter_whiteboard     |           |        |          |         |
+--------------------------+-----------+--------+----------+---------+
| slide                    |           |        |          |         |
+--------------------------+-----------+--------+----------+---------+
| whiteboard               |           |        |          |         |
+--------------------------+-----------+--------+----------+---------+

+--------------+-----------+--------+----------+---------+
| Metric       | Precision | Recall | F1-Score | Support |
+==============+===========+========+==========+=========+
| accuracy     |           |        |          |         |
+--------------+-----------+--------+----------+---------+
| macro avg    |           |        |          |         |
+--------------+-----------+--------+----------+---------+
| weighted avg |           |        |          |         |
+--------------+-----------+--------+----------+---------+