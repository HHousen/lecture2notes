Dataset General Information
===========================

Directory Structure
-------------------

* **classifier-data**: Created by :ref:`ss_compile_data`. Contains all extracted slides and extracted sorted frames from the videos directory. This is the folder that should be given to the model for training.
* **scraper-scripts**: Contains all of the scripts needed to obtain and manipulate the data. See :ref:`ss_home` for more information.
* **slides**:
    * *images*: The location where slide images extracted from slideshows in *pdfs* subdirectory are saved (used by :ref:`ss_pdf2image`).
    * *pdfs*: The location where downloaded slideshow PDFs are saved (used by :ref:`ss_slides_downloader`).
* **videos**: Contains the following directory structure for each downloaded video:
    * `video_id`: The parent folder containing all the files related to the specific video.
        * frames: All frames extracted from `video_id` by :ref:`ss_frame_extractor`.
        * frames_sorted: Frames from `video_id` that are grouped into correct classes. :ref:`ss_auto_sort` can help with this but you must verify correctness. More at :ref:`ss_auto_sort`.
* **slides-dataset.csv**: A list of all the slide presentations used in the dataset. **NOT** automatically updated by :ref:`ss_slides_downloader`. You must manually update this file if you want the dataset to be reproducible.
* **sort_file_map.csv**: A list of filenames and categories. Used exclusively by :ref:`ss_sort_from_file` to either ``make`` a file mapping of the category to which each frame belongs or to ``sort`` each file in ``sort_file_map.csv``, moving the respective frame from ``video_id/frames`` to ``video_id/frames_sorted/category``.
* **to-be-sorted.csv**: A list of videos and specific frames that have been sorted by :ref:`ss_auto_sort` but need to be checked by a human for correctness. When running :ref:`ss_auto_sort` any frames where the AI model's confidence level is below a threshold are added to this list as most likely incorrect.
* **videos-dataset.csv**: A list of all videos used in the dataset. Automatically updated by :ref:`ss_youtube_scraper` and :ref:`ss_website_scraper`. The `provider` column is used by :ref:`ss_video_downloader` to determine how to download the video.

Walkthrough (Step-by-Step Instructions to Create Dataset)
---------------------------------------------------------

1. Download Content:
    1. Download all videos: ``python 2-video_downloader.py csv``
    2. Download all slides: ``python 2-slides_downloader.py csv``
2. Data Pre-processing:
    1. Convert slide PDFs to PNGs: ``python 3-pdf2image.py``
    2. Extract frames from all videos: ``python 3-frame_extractor.py auto``
    3. Sort the frames: ``python 4-sort_from_file.py sort``
3. Compile and merge the data: ``python 5-compile_data.py``