# lecture2notes

> Research in progress. Code for the entirety of [Research Paper Name]. Converts a lecture presentation to detailed notes.

Brief Summary: A 

### Notes

While this project was designed to process presentations with slides, it will work if there are no slides, but only audio will be used for summarization.

Some code supports fastai since that was the library that was initially used to train `Models/slide-classifier`. However, certain sections do not support fastai (most noteably is [Models/slide-classifier/inference.py](Models/slide-classifier/inference.py) which feature extraction component that is necessary for clustering). For this reason, it is recommended to use only pytorch, although the sections that support fastai should work (but have not been tested in a while).

## Install
Installation is made easy due to conda environments. Simply run this command from the root project directory: `conda env create` and conda will create and environment called `lecture2notes` with all the required packages in [environment.yml](environment.yml).

NOTE: The `srt` package is on pypi but not conda. Therefore you have to build it locally. More details can be found in the [conda documentation](https://docs.conda.io/projects/conda-build/en/latest/user-guide/tutorials/build-pkgs-skeleton.html). This can de done by performing the following steps:
1. `conda skeleton pypi srt`
2. Open the `srt` folder that was created in your workign directory and edit the `meta.yaml` file `requirements` section so that python version 3.6 is used. Change `python` under the `host` and `run` sections to `python 3.6`.
3. `conda build srt`
4. You're done. The `srt` package is now ready to be installed from the `local` channel. You can now create the conda environment using `conda env create`.

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
* **sort_file_map.csv**: A list of filenames and categories. Used exclusively by [scraper-scripts/4-sort_from_file.py](Dataset/scraper-scripts/4-sort_from_file.py) to either `make` a file mapping of the category to which each frame belongs or to `sort` each file in [sort_file_map.csv](Dataset/sort_file_map.csv), moving the respective frame from `video_id/frames` to `video_id/frames_sorted/category`.
* **to-be-sorted.csv**: A list of videos and specific frames that have been sorted by [scraper-scripts/4-auto_sort.py](Dataset/scraper-scripts/4-auto_sort.py) but need to be checked by a human for correctness. When running [scraper-scripts/4-auto_sort.py](Dataset/scraper-scripts/4-auto_sort.py) any frames where the AI model's confidence level is below a threshold are added to this list as most likely incorrect.
* **videos-dataset.csv**: A list of all videos used in the dataset. Automatically updated by [scraper-scripts/1-youtube_scraper.py](Dataset/scraper-scripts/1-youtube_scraper.py) and [scraper-scripts/1-website_scraper.py](Dataset/scraper-scripts/1-website_scraper.py). The `provider` column is used to determine how to download the video in [scraper-scripts/2-video_downloader.py](Dataset/scraper-scripts/2-video_downloader.py).

#### Script Descriptions
> All scripts that are needed to obtain and manipulate the data. Located in `Dataset/scraper-scripts`

* **website_scraper**: Takes a video page link, video download link, and video published date and then adds that information to [videos-dataset.csv](Dataset/videos-dataset.csv).
    * Command: `python 1-website_scraper.py <date> <page_link> <video_download_link> <description (optional)>`
    * Examples:
        * `python 1-website_scraper.py 1-1-2010 https://oyc.yale.edu/astronomy/astr-160/update-1 http://openmedia.yale.edu/cgi-bin/open_yale/media_downloader.cgi?file=/courses/spring07/astr160/mov/astr160_update01_070212.mov`
* **youtube_scraper**: Takes a video id or channel id from youtube, extracts important information using the YouTube Data API, and then adds that information to [videos-dataset.csv](Dataset/videos-dataset.csv).
    * Output of `python 1-youtube_scraper.py --help`:
    ```
    usage: 1-youtube_scraper.py [-h] [-n N] [-t] {video,channel,transcript} STR

    YouTube Scraper

    positional arguments:
    {video,channel,transcript}
                            Get metadata for a video or a certain number of videos
                            from a channel. Transcript mode downloads the
                            transcript for a video_id.
    STR                   Channel or video id depending on mode

    optional arguments:
    -h, --help            show this help message and exit
    -n N, --num_pages N   Number of pages of videos to scape if mode is
                            `channel`
    -t, --transcript      Download transcript for each video scraped.
    ```
* **slides_downloader**: Takes a link to a pdf slideshow and downloads it to `Dataset/slides/pdfs` or downloads every entry in [slides-dataset.csv](Dataset/slides-dataset.csv).
    * Command: `python slides_downloader.py <csv/your_url>`
    * Examples:
        * If *csv*: `python 2-slides_downloader.py csv`
        * If *your_url*: `python 2-slides_downloader.py https://ocw.mit.edu/courses/history/21h-343j-making-books-the-renaissance-and-today-spring-2016/lecture-slides/MIT21H_343JS16_Print.pdf`
    * Required Software: `wget`
* **video_downloader**: Uses youtube-dl (for `youtube` videos) and wget (for `website` videos) to download either a youtube video by id or every video that has not been download in [videos-dataset.csv](Dataset/videos-dataset.csv).
    * Command: `python 2-video_downloader.py <csv/youtube your_youtube_video_id>`
    * Examples:
        * If *csv*: `python 2-video_downloader.py csv`
        * If *your_youtube_video_id*: `python 2-video_downloader.py youtube 1Qws70XGSq4`
    * Required Software: `youtube-dl` ([Github](https://github.com/ytdl-org/youtube-dl)/[Website](https://ytdl-org.github.io/youtube-dl/index.html)), `wget`
* **frame_extractor**: Extracts either every n frames from a video file (selected by id and must be in `videos` folder) or, in `auto` mode, every n frames from every video in the dataset that has been downloaded and has not had its frames extracted already. `extract_every_x_seconds` can be set to auto to use the `get_extract_every_x_seconds()` function to automatically determine a good number of frames to extract. `auto` mode uses this feature and allows for exact reconstruction of the dataset. Extracted frames are saved into `Dataset/videos/video_id/frames`.
    * Command: `python 3-frame_extractor.py <video_id/auto> <extract_every_x_seconds/auto> <quality>`
    * Examples:
        * If *video_id*: `python 3-frame_extractor.py VT2o4KCEbes 20 5` or to automatically extract a good number of frames: `python 3-frame_extractor.py 63hAHbkzJG4 auto 5`
        * If *auto*:  `python 3-frame_extractor.py auto`
    * Required Software: `ffmpeg` ([Github](https://github.com/FFmpeg/FFmpeg)/[Website](https://www.ffmpeg.org/))
* **pdf2image**: Takes every page in all pdf files in `Dataset/slides/pdfs`, converts them to png images, and saves them in `Dataset/slides/images/pdf_file_name`. Requires `pdftoppm` package. 
    * Command: `python 3-pdf2image.py`
    * Required Software: `poppler-utils (pdftoppm)` ([Man Page](https://linux.die.net/man/1/pdftoppm)/[Website](https://poppler.freedesktop.org/))
* **auto_sort**: Goes through every extracted frame for all videos in the dataset that don’t have sorted frames and classifies them using `Models/slide-classifier`. You need either a trained fastai or pytorch model to use this. This code works with both fastai and pytorch models. Creates a list of frames that need to be checked for correctness by humans in [to-be-sorted.csv](Dataset/to-be-sorted.csv). This also requires certain files from `Models/slide-classifier`.
    * Command: `python 4-auto_sort.py <fastai/pytorch>`
* **sort_from_file**: Either `make` a file mapping of the category to which each frame belongs or to `sort` each file in [sort_file_map.csv](Dataset/sort_file_map.csv), moving the respective frame from `video_id/frames` to `video_id/frames_sorted/category`. Only purpose is to exactly reconstruct the dataset without downloading already sorted images.
    * Command: `python 4-sort_from_file.py <make/sort>`
* **compile_data**: Merges the sorted frames from all the videos in the dataset to `Dataset/classifier-data`.
    * Command: `python 5-compile_data.py`

#### Walkthrough (Step-by-Step Instructions to Create Dataset)

> Either download compiled dataset from [Dataset Direct Download Link] or use the following steps.

1. Install Prerequisite Software
    * wget: `apt install wget`
    * youtube-dl (most up-to-date instructions on [their website](https://ytdl-org.github.io/youtube-dl/index.html)):
    ```bash
    sudo curl -L https://yt-dl.org/downloads/latest/youtube-dl -o /usr/local/bin/youtube-dl

    sudo chmod a+rx /usr/local/bin/youtube-dl
    ```
    * ffmpeg: `apt install ffmpeg`
    * pdftoppm: `apt install poppler-utils`
2. Download Content:
    1. Download all videos: `python 2-video_downloader.py csv`
    2. Download all slides: `python 2-slides_downloader.py csv`
3. Data Pre-processing:
    1. Convert slide pdfs to pngs: `python 3-pdf2image.py`
    2. Extract frames from all videos: `python 3-frame_extractor.py auto`
    3. Sort the frames: `python 4-sort_from_file.py sort`
4. Compile and merge the data: `python 5-compile_data.py`

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
    * Output of `python main.py --help`:
    ```
    usage: main.py [-h] [-s N] [-d PATH] [-id] [-rm] [-c]
            [-tm {sphinx,google,youtube}] [--video_id ID]
            DIR

    End-to-End Conversion of Lecture Videos to Notes using ML

    positional arguments:
    DIR                   path to video

    optional arguments:
    -h, --help            show this help message and exit
    -s N, --skip-to N     set to > 0 to skip specific processing steps
    -d PATH, --process_dir PATH
                            path to the proessing directory (where extracted
                            frames and other files are saved), set to "automatic"
                            to use the video's folder (default: ./)
    -id, --auto-id        automatically create a subdirectory in `process_dir`
                            with a unique id for the video and change
                            `process_dir` to this new directory
    -rm, --remove         remove `process_dir` once conversion is complete
    -c, --chunk           split the audio into small chunks on silence
    -tm {sphinx,google,youtube}, --transcription_method {sphinx,google,youtube}
                            specify the program that should be used for
                            transcription. Either CMU Sphinx (works offline) or
                            Google Speech Recognition (probably will require
                            chunking) or pull a video transcript from YouTube
                            based on video_id
    --video_id ID         id of youtube video to get subtitles from
    ```
* **ocr**: Provides `all_in_folder()`, which performs OCR on every file in folder and returns results, and `write_to_file`, which writes everything stored in `results` to file at path `save_file` (used to write results from `all_in_folder()` to `save_file`).

## Meta

Hayden Housen – [haydenhousen.com](https://haydenhousen.com)

Distributed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) for more information.

<https://github.com/HHousen>

## Contributing

1. Fork it (<https://github.com/HHousen/lecture2notes/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request
