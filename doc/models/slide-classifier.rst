Slide Classifier
================

.. _sc_overview:

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

.. note:: Visit the :ref:`slide_classifier_api` page to see the documentation for each function in more detail.

* **class_cluster_scikit.py**: Implements ``KMeans`` and ``AffinityPropagation`` from ``sklearn.cluster`` to provde a ``Cluster`` class. The code is documented in file. The purpose is to add feature vectors using ``add()``, then cluster the features, and finally return a list of files and their corresponding cluster centroids with ``create_move_list()``. Three important functions and their use cases follow:

    * ``create_move_list()`` is called by ``end_to_end/cluster.py`` and returns a list of filenames and their corresponding clusters.
    * ``calculate_best_k()`` generates a graph (saved to ``best_k_value.png`` if using Agg matplotlib backend) that graphs the cost (squared error) as a function of the number of centroids (value of k) if the algorithm is ``"kmeans"``. The point at which the graph becomes essentially linear is the optimal value of k.
    * ``visualize()`` creates a tensorboard projection of the cluster for simplified viewing and understanding.

* **class_cluster_faiss.py**: An outdated version of **class_cluster_scikit** that uses `facebookresearch/faiss <https://github.com/facebookresearch/faiss>`_ (specifically the kmeans implementation `documented here <https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks:-clustering,-PCA,-quantization>`_) to provide a ``Cluster`` class. More details in the ``class_cluster_scikit`` entry above.
* **custom_nnmodules.py**: Provides a few custom (copied from `fastai <https://github.com/fastai/fastai>`_) nn.Modules.
* **inference.py**: Sets up model and provides ``get_prediction()``, which takes an image and returns a prediction and extracted features. 
* **lr_finder.py**: Slightly modified (allows usage of matplotlib Agg backend) code from `davidtvs/pytorch-lr-finder <https://github.com/davidtvs/pytorch-lr-finder>`_ to find the best learning rate.
* **mish.py**: Code for the mish activation function.
* **slide-classifier-fastai.ipynb**: Notebook to train simple fastai classifier on the dataset in ``dataset/classifier-data``. It is outdated and not supported and only remains in the repository as an example.
* **slide_classifier_helpers.py**: Helper functions for ``slide_classifier_pytorch.py``. Includes RELU to Mish activation function conversion and confusion matrix plotting functions among others.
* **slide_classifier_pytorch.py**: The main model code which uses advanced features such as the AdamW optimizer and a modified ResNet that allows for more effective pre-training/feature extracting.
* **slide-classifier-pytorch-old.py**: The old version of the slide classifier model training code. This old version was not organized as well as the current version. The old version was raw PyTorch code since it did not utilize ``pytorch_lightning``.

Experiments
-----------

Tested the `EfficientNet <https://arxiv.org/abs/1905.11946>`_ vs `Resnet34 <https://arxiv.org/abs/1512.03385>`_ architectures and `AdamW <https://pytorch.org/docs/stable/optim.html#torch.optim.AdamW>`_ vs `Ranger <https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer>`_ optimizers with and without the `OneCycle scheduler <https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.OneCycleLR>`_.

.. important:: Interactive charts, graphs, raw data, run commands, hyperparameter choices, and more for all experiments are publicly available on the `Lecture2Notes-Slide_Classifier Weights & Biases page <https://app.wandb.ai/hhousen/lecture2notes-slide_classifier>`_.


.. _slide_classifier_main_results:

Main Results
^^^^^^^^^^^^

The section contains the high-level graphs and main takeaways. To see the individual results for each experiment in table form scroll to :ref:`slide_classifier_training_configs`.

.. _slide_classifier_training_configs:

Training Configurations
^^^^^^^^^^^^^^^^^^^^^^^

The commands and individual results in table form for each model trained. Scroll to :ref:`slide_classifier_main_results` to see the main takeaways.

1. **efficientnet-b0_ranger_onecycle:**

    Command:

    .. code-block:: bash

        python slide_classifier_pytorch.py ../../dataset/classifier-data-train-val/ --do_test --do_train --max_epochs 10 --seed 42 --pretrained --arch efficientnet-b0 --use_scheduler onecycle --optimizer ranger --ranger_k 3 --momentum 0.95 --optimizer_eps 1e-5
    
    Results:

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

2. **efficientnet-b0_adamw_onecycle:**

    .. code-block:: bash

        slide_classifier_pytorch.py ../../dataset/classifier-data/ --random_split --do_test --do_train --max_epochs 10 --seed 42 --pretrained --arch efficientnet-b0 --use_scheduler onecycle

3. **efficientnet-b0_ranger:**

    .. code-block:: bash

        python slide_classifier_pytorch.py ../../dataset/classifier-data/ --random_split --do_test --do_train --max_epochs 10 --seed 42 --pretrained --arch efficientnet-b0 --optimizer ranger --ranger_k 3 --momentum 0.95 --optimizer_eps 1e-5

4. **efficientnet-b0_adamw:**

    .. code-block:: bash

        slide_classifier_pytorch.py ../../dataset/classifier-data/ --random_split --do_test --do_train --max_epochs 10 --seed 42 --pretrained --arch efficientnet-b0

5. **resnet34_ranger_onecycle:**

    .. code-block:: bash

        python slide_classifier_pytorch.py ../../dataset/classifier-data/ --random_split --do_test --do_train --max_epochs 10 --seed 42 --pretrained --arch resnet34 --use_scheduler onecycle --optimizer ranger --ranger_k 3 --momentum 0.95 --optimizer_eps 1e-5

6. **resnet34_adamw_onecycle:**

    .. code-block:: bash

        slide_classifier_pytorch.py ../../dataset/classifier-data/ --random_split --do_test --do_train --max_epochs 10 --seed 42 --pretrained --arch resnet34 --use_scheduler onecycle

7. **resnet34_ranger:**

    .. code-block:: bash

        python slide_classifier_pytorch.py ../../dataset/classifier-data/ --random_split --do_test --do_train --max_epochs 10 --seed 42 --pretrained --arch resnet34 --optimizer ranger --ranger_k 3 --momentum 0.95 --optimizer_eps 1e-5

8. **resnet34_adamw:**

    .. code-block:: bash

        slide_classifier_pytorch.py ../../dataset/classifier-data/ --random_split --do_test --do_train --max_epochs 10 --seed 42 --pretrained --arch resnet34

Slide-Classifier-Pytorch Help
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Output of ``python slide_classifier_pytorch.py --help``:

.. code-block:: bash

    usage: slide_classifier_pytorch.py [-h] [--default_root_dir DEFAULT_ROOT_DIR]
                                        [--min_epochs MIN_EPOCHS]
                                        [--max_epochs MAX_EPOCHS]
                                        [--min_steps MIN_STEPS]
                                        [--max_steps MAX_STEPS] [--lr LR]
                                        [--check_val_every_n_epoch CHECK_VAL_EVERY_N_EPOCH]
                                        [--gpus GPUS] [--overfit_pct OVERFIT_PCT]
                                        [--train_percent_check TRAIN_PERCENT_CHECK]
                                        [--val_percent_check VAL_PERCENT_CHECK]
                                        [--test_percent_check TEST_PERCENT_CHECK]
                                        [--amp_level AMP_LEVEL]
                                        [--precision PRECISION] [--seed SEED]
                                        [--profiler]
                                        [--progress_bar_refresh_rate PROGRESS_BAR_REFRESH_RATE]
                                        [--num_sanity_val_steps NUM_SANITY_VAL_STEPS]
                                        [--use_logger {tensorboard,wandb}]
                                        [--do_train] [--do_test]
                                        [--load_weights LOAD_WEIGHTS]
                                        [--load_from_checkpoint LOAD_FROM_CHECKPOINT]
                                        [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                                        [-a ARCH] [-j N]
                                        [--train_batch_size TRAIN_BATCH_SIZE]
                                        [--val_batch_size VAL_BATCH_SIZE]
                                        [--test_batch_size TEST_BATCH_SIZE]
                                        [--momentum M] [--weight_decay W] [-k K]
                                        [--optimizer_alpha N] [--optimizer_eps N]
                                        [--pretrained] [--random_split]
                                        [--relu_to_mish]
                                        [--feature_extract {normal,advanced,none}]
                                        [-o OPTIMIZER]
                                        DIR

        positional arguments:
        DIR                   path to dataset

        optional arguments:
        -h, --help            show this help message and exit
        --default_root_dir DEFAULT_ROOT_DIR
                                Default path for logs and weights
        --min_epochs MIN_EPOCHS
                                Limits training to a minimum number of epochs
        --max_epochs MAX_EPOCHS
                                Limits training to a max number number of epochs
        --min_steps MIN_STEPS
                                Limits training to a minimum number number of steps
        --max_steps MAX_STEPS
                                Limits training to a max number number of steps
        --lr LR, --learning_rate LR
                                initial learning rate
        --check_val_every_n_epoch CHECK_VAL_EVERY_N_EPOCH
                                Check val every n train epochs.
        --gpus GPUS           Number of GPUs to train on or Which GPUs to train on.
                                (default: -1 (all gpus))
        --overfit_pct OVERFIT_PCT
                                Uses this much data of all datasets (training,
                                validation, test). Useful for quickly debugging or
                                trying to overfit on purpose.
        --train_percent_check TRAIN_PERCENT_CHECK
                                How much of training dataset to check. Useful when
                                debugging or testing something that happens at the end
                                of an epoch.
        --val_percent_check VAL_PERCENT_CHECK
                                How much of validation dataset to check. Useful when
                                debugging or testing something that happens at the end
                                of an epoch.
        --test_percent_check TEST_PERCENT_CHECK
                                How much of test dataset to check.
        --amp_level AMP_LEVEL
                                The optimization level to use (O1, O2, etc…) for
                                16-bit GPU precision (using NVIDIA apex under the
                                hood).
        --precision PRECISION
                                Full precision (32), half precision (16). Can be used
                                on CPU, GPU or TPUs.
        --seed SEED           Seed for reproducible results. Can negatively impact
                                performace in some cases.
        --profiler            To profile individual steps during training and assist
                                in identifying bottlenecks.
        --progress_bar_refresh_rate PROGRESS_BAR_REFRESH_RATE
                                How often to refresh progress bar (in steps). In
                                notebooks, faster refresh rates (lower number) is
                                known to crash them because of their screen refresh
                                rates, so raise it to 50 or more.
        --num_sanity_val_steps NUM_SANITY_VAL_STEPS
                                Sanity check runs n batches of val before starting the
                                training routine. This catches any bugs in your
                                validation without having to wait for the first
                                validation check.
        --use_logger {tensorboard,wandb}
                                Which program to use for logging.
        --do_train            Run the training procedure.
        --do_test             Run the testing procedure.
        --load_weights LOAD_WEIGHTS
                                Loads the model weights from a given checkpoint
        --load_from_checkpoint LOAD_FROM_CHECKPOINT
                                Loads the model weights and hyperparameters from a
                                given checkpoint.
        -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --log {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                                Set the logging level (default: 'Info').
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
        --train_batch_size TRAIN_BATCH_SIZE
                                Batch size per GPU/CPU for training.
        --val_batch_size VAL_BATCH_SIZE
                                Batch size per GPU/CPU for evaluation.
        --test_batch_size TEST_BATCH_SIZE
                                Batch size per GPU/CPU for testing.
        --momentum M          momentum. Ranger optimizer suggests 0.95.
        --weight_decay W      weight decay (default: 1e-2)
        -k K, --ranger_k K    Ranger (LookAhead) optimizer k value (default: 6)
        --optimizer_alpha N   Optimizer alpha parameter (default: 0.999)
        --optimizer_eps N     Optimizer eps parameter (default: 1e-8)
        --pretrained          use pre-trained model
        --random_split        use random_split to create train and val set instead
                                of train and val folders
        --relu_to_mish        convert any relu activations to mish activations
        --feature_extract {normal,advanced,none}
                                If `False` or `None`, finetune the whole model. When
                                `normal`, only update the reshaped layer params. When
                                `advanced`, use fastai version of feature extracting
                                (add fancy group of layers and only update this group
                                and BatchNorm)
        -o OPTIMIZER, --optimizer OPTIMIZER
                                Optimizer to use (default=AdamW)