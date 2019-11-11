# lecture2notes

> Research in progress. Code for the entirety of [Research Paper Name]. Converts a lecture presentation to detailed notes.

Brief Summary: A 

### Notes

While this project was designed to process presentations with slides, it will work if there are no slides, but only audio will be used for summarization.

Some code supports fastai since that was the library that was initially used to train `Models/slide-classifier`. However, certain sections do not support fastai (most noteably is [Models/slide-classifier/inference.py](Models/slide-classifier/inference.py) which feature extraction component that is necessary for clustering). For this reason, it is recommended to use only pytorch, although the sections that support fastai should work (but have not been tested in a while).

## Install
Installation is made easy due to conda environments. Simply run this command from the root project directory: `conda env create` and conda will create and environment called `lecture2notes` with all the required packages in [environment.yml](environment.yml).

### Step-by-Step Instructions
1. Clone this repository: `git clone https://github.com/HHousen/lecture2notes.git`.
2. Change to project diretory: `cd lecture2notes`.
3. Run installation command: `conda env create`.

## Components

### Dataset

> The code used to compile the data needed to train Models/slide-classifier

#### Folder Structure

* **classifier-data**: Created by [scraper-scripts/5-compile_data.py](Dataset/scraper-scripts/5-compile_data.py). Contains all extracted slides and extracted sorted frames from the slides and videos directories. This is the folder that should be given to the model for training.
* **scraper-scripts**: Contains all of the scripts needed to obtain and manipulate the data.
* **slides**: 
    * *pdfs* subdirectory: Used by [scraper-scripts/2-slides_downloader.py](Dataset/scraper-scripts/2-slides_downloader.py) as the location to save downloaded slideshow pdfs.
    * *images* subdirectory: Used by [scraper-scripts/3-pdf2image.py](Dataset/scraper-scripts/3-pdf2image.py) as the location to save slide images extracted from slideshows in *pdfs* subdirectory.
* **videos**: Contains the following directory structure for each downloaded video:
    * `video_id`: The parent folder containing all the files related to the specific video.
        * frames: All frames extracted from `video_id` by [scraper-scripts/3-frame_extractor.py](Dataset/scraper-scripts/3-frame_extractor.py)
        * frames_sorted: Frames from `video_id` that are grouped into correct classes. [scraper-scripts/4-auto_sort.py](Dataset/scraper-scripts/4-auto_sort.py) can help with this but you must verify correctness. More info in *Script Descriptions*.
* **slides-dataset.csv**: A list of all slides used in the dataset. **NOT** automatically updated by [scraper-scripts/2-slides_downloader.py](Dataset/scraper-scripts/2-slides_downloader.py). You need to manually update this file if you want the dataset to be reproducible.
* **to-be-sorted.csv**: A list of videos and specific frames that have been sorted by [scraper-scripts/4-auto_sort.py](Dataset/scraper-scripts/4-auto_sort.py) but need to be checked by a human for correctness. When running [scraper-scripts/4-auto_sort.py](Dataset/scraper-scripts/4-auto_sort.py) any frames where the AI model's confidence level is below a threshold are added to this list as most likely incorrect.
* **videos-dataset.csv**: A list of all videos used in the dataset. Automatically updated by [scraper-scripts/1-youtube_scraper.py](Dataset/scraper-scripts/1-youtube_scraper.py).

#### Script Descriptions
> All scripts that are needed to obtain and manipulate the data. Located in `Dataset/scraper-scripts`

* **youtube_scraper**: Takes a video id or channel id from youtube, extracts important information using the YouTube Data API, and then adds that information to [slides-dataset.csv](Dataset/videos-dataset.csv).
    * Command: `python 1-youtube_scraper.py <channel/video> <video_id/channel_id> <number of pages (only if channel)>`
    * Examples:
        * If *channel*: `python 1-youtube_scraper.py channel UCEBb1b_L6zDS3xTUrIALZOw 10`
        * If *video*: `python 1-youtube_scraper.py video 1Qws70XGSq4`
* **slides_downloader**: Takes a link to a pdf slideshow and downloads it to `Dataset/slides/pdfs` or downloads every entry in [slides-dataset.csv](Dataset/slides-dataset.csv).
    * Command: `python slides_downloader.py <csv/your_url>`
    * Examples:
        * If *csv*: `python 2-slides_downloader.py csv`
        * If *your_url*: `python 2-slides_downloader.py https://ocw.mit.edu/courses/history/21h-343j-making-books-the-renaissance-and-today-spring-2016/lecture-slides/MIT21H_343JS16_Print.pdf`
* **youtube_downloader**: Uses youtube-dl to download either a video by id or every video that has not been download in [videos-dataset.csv](Dataset/videos-dataset.csv).
    * Command: `python 2-youtube_downloader.py <csv/your_youtube_video_id>`
    * Examples:
        * If *csv*: `python 2-youtube_downloader.py csv`
        * If *your_youtube_video_id*: `python 2-youtube_downloader.py 1Qws70XGSq4`
* **frame_extractor**: Extracts either every n frames from a video file (selected by id and must be in `videos` folder) or, in `auto` mode, every n frames from every video in the dataset that has been downloaded and has not had its frames extracted already. `extract_every_x_seconds` can be set to auto to use the `get_extract_every_x_seconds()` function to automatically determine a good number of frames to extract. `auto` mode uses this feature and allows for exact reconstruction of the dataset. Extracted frames are saved into `Dataset/videos/video_id/frames`.
    * Command: `python 3-frame_extractor.py <video_id/auto> <extract_every_x_seconds/auto> <quality>`
    * Examples:
        * If *video_id*: `python 3-frame_extractor.py VT2o4KCEbes 20 5` or to automatically extract a good number of frames: `python 3-frame_extractor.py 63hAHbkzJG4 auto 5`
        * If *auto*:  `python 3-frame_extractor.py auto`
* **pdf2image**: Takes every page in all pdf files in `Dataset/slides/pdfs`, converts them to png images, and saves them in `Dataset/slides/images/pdf_file_name`. Requires `pdftoppm` package. 
    * Command: `python 3-pdf2image.py`
* **auto_sort**: Goes through every extracted frame for all videos in the dataset that don’t have sorted frames and classifies them using `Models/slide-classifier`. You need either a trained fastai or pytorch model to use this. This code works with both fastai and pytorch models. Creates a list of frames that need to be checked for correctness by humans in [to-be-sorted.csv](Dataset/to-be-sorted.csv). This also requires certain files from `Models/slide-classifier`.
    * Command: `python 4-auto_sort.py <fastai/pytorch>`
* **compile_data**: Merges the sorted frames from all the videos in the dataset to `Dataset/classifier-data`.
    * Command: `python 5-compile_data.py`

#### Walkthrough (Step-by-Step Instructions to Create Dataset)

> Either download compiled dataset from [Dataset Direct Download Link] or use the following steps.

1. Download Content:
    1. Download all videos: `python 2-youtube_downloader.py csv`
    2. Download all slides: `python 2-slides_downloader.py csv`
2. Data Pre-processing:
    1. Convert slide pdfs to pngs: `python 3-pdf2image.py`
    2. Extract frames from all videos: `python 3-frame_extractor.py auto`
    3. Auto sort the frames: `python 4-auto_sort.py pytorch` and check for correctness
3. Compile and merge the data: `python 5-compile_data.py`

### Models

> All of the machine learning model code used in this project is located here.

* **class_cluster_scikit**: Implements `KMeans` and `AffinityPropagation` from `sklearn.cluster` to provde a `Cluster` class. Code is documented in file. Purpose is to add feature vectors using `add()`, then cluster the features, and finally return a list of files and their corresponding cluster centroids with `create_move_list()`. Two important functions and their use cases follow:
    * `create_move_list()` function is what is called by [cluster.py](End-To-End/cluster.py) and returns a list of filenames and their coresponding clusters.
    * `calculate_best_k()` function generates a graph (saved to `best_k_value.png` if using Agg matplotlib backend) that graphs the cost (squared error) as a function of the number of centroids (value of k) if the algorithm is `"kmeans"`.
* **class_cluster_faiss**: An outdated version of [class_cluster_scikit.py](Models/slide-classifier/class_cluster_scikit.py) that uses [facebookresearch/faiss](https://github.com/facebookresearch/faiss) (specifically the kmeans implementation documented [here](https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks:-clustering,-PCA,-quantization)) to provide a `Cluster` class. More details in the `class_cluster_scikit` entry of this section of the documentation.
* **custom_nnmodules**: Provides a few custom (copied from [fastai](https://github.com/fastai/fastai)) nn.Modules.
* **inference**: Sets up model and provides `get_prediction()`, which takes an image and returns a prediction and extracted features. 
* **lr_finder**: Slightly modified (allows usage of matplotlib Agg backend) code from [davidtvs/pytorch-lr-finder](https://github.com/davidtvs/pytorch-lr-finder)
* **slide-classifier-fastai.ipynb**: Notebook to train simple fastai classifier on the dataset in `Dataset/classifier-data`.
* **slide-classifier-pytorch.py**: The main model code which is written completely in PyTorch and uses advanced features such as the AdamW optimizer and a modified ResNet that allows for more effective pretraining/feature extracting.
    * Output of `python slide-classifier-pytorch.py --help`:
    ```
    usage: slide-classifier-pytorch.py [-h] [-a ARCH] [-j N] [--epochs N]
                                   [--start-epoch N] [-b N] [--lr LR]
                                   [--momentum M] [--wd W] [-p N]
                                   [--resume PATH] [-e] [--pretrained]
                                   [--seed SEED] [--gpu GPU] [--random_split]
                                   [--feature_extract {normal,advanced}]
                                   [--find_lr] [-o OPTIM]
                                   DIR

    PyTorch Slide Classifier Training

    positional arguments:
    DIR                   path to dataset

    optional arguments:
    -h, --help            show this help message and exit
    -a ARCH, --arch ARCH  model architecture: alexnet | densenet121 |
                            densenet161 | densenet169 | densenet201 | inception_v3
                            | resnet101 | resnet152 | resnet18 | resnet34 |
                            resnet50 | squeezenet1_0 | squeezenet1_1 | vgg11 |
                            vgg11_bn | vgg13 | vgg13_bn | vgg16 | vgg16_bn | vgg19
                            | vgg19_bn (default: resnet34)
    -j N, --workers N     number of data loading workers (default: 4)
    --epochs N            number of total epochs to run (default: 6)
    --start-epoch N       manual epoch number (useful on restarts)
    -b N, --batch-size N  mini-batch size (default: 16)
    --lr LR, --learning-rate LR
                            initial learning rate
    --momentum M          momentum
    --wd W, --weight-decay W
                            weight decay (default: 1e-2)
    -p N, --print-freq N  print frequency (default: 10)
    --resume PATH         path to latest checkpoint (default: none)
    -e, --evaluate        evaluate model on validation set
    --pretrained          use pre-trained model
    --seed SEED           seed for initializing training.
    --gpu GPU             GPU id to use.
    --random_split        use random_split to create train and val set instead
                            of train and val folders
    --feature_extract {normal,advanced}
                            If False, we finetune the whole model. When normal we
                            only update the reshaped layer params. When advanced
                            use fastai version of feature extracting (add fancy
                            group of layers and only update this group and
                            BatchNorm)
    --find_lr             Flag for lr_finder.
    -o OPTIM, --optimizer OPTIM
                            Optimizer to use (default=AdamW)
    ```

### End-To-End

> The end-to-end approach. One command to take a video file and return summarized notes.

Run `python main.py <path_to_video>` to get a notes file.

#### Scripts Descriptions

* **cluster**: Provides `make_clusters()`, which clusters all images in directory `slides_dir` and saves each cluster to a subfolder in `cluster_dir` (directory in parent of `slides_dir`).
* **frames_extractor**: Provides `extract_frames()`, which extracts frames from `input_video_path` at quality level `quality` (best quality is 2) every `extract_every_x_seconds seconds` and saves them to `output_path`.
* **helpers**: A small file of helper functions to reduce duplicate code.
* **main**: The master file that brings all of the components in this directory together by calling functions provided by the components with correct parameters. Implements a `skip_to` variable that can be set to skip to a certain step of the process. This is useful if a pervious step completed but the overall process failed. If this happens the `skip_to` variable allows the user to fix the problem and resume from the point where the error occurred.
    * Example: `python main.py <path_to_video>`
    * Example with *skip_to*: `python main.py <path_to_video> <skip_to>`
* **ocr**: Provides `all_in_folder()`, which performs OCR on every file in folder and returns results, and `write_to_file`, which writes everything stored in `results` to file at path `save_file` (used to write results from `all_in_folder()` to `save_file`).

## Meta

Hayden Housen – [haydenhousen.com](https://haydenhousen.com)

Distributed under the MIT license. See the [LICENSE](LICENSE) for more information.

<https://github.com/HHousen>

## Contributing

1. Fork it (<https://github.com/HHousen/lecture2notes/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request
