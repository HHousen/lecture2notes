.. _dataset_general_information:

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

.. _dataset_general_walkthrough:

Walkthrough (Step-by-Step Instructions to Create Dataset)
---------------------------------------------------------

1. Install Prerequisite Software: ``youtube-dl``, ``wget``, ``ffmpeg``, ``poppler-utils`` (see :ref:`quick_install`)
2. Download Content:
    1. Download all videos: ``python 2-video_downloader.py csv``
    2. Download all slides: ``python 2-slides_downloader.py csv``
3. Data Pre-processing:
    1. Convert slide PDFs to PNGs: ``python 3-pdf2image.py``
    2. Extract frames from all videos: ``python 3-frame_extractor.py auto``
    3. Sort the frames: ``python 4-sort_from_file.py sort``
4. Compile and merge the data: ``python 5-compile_data.py``

Transcripts WER
---------------

Script location: ``dataset/transcripts_wer.py``

This script will calculate the Word Error Rate (WER), Match Error Rate (MER), and Word Information Lost (WIL) for all videos in ``dataset/videos-dataset.csv`` that are YouTube videos with manual transcripts added (see the :ref:`YouTube transcription method <transcribe_methods>` for more info about transcripts on YouTube).

There are two modes:

1. ``transcribe``: Runs speech-to-text with DeepSpeech.

    Process: For each transcript in ``dataset/transcripts``:

    1. Download the audio for the video
    2. Convert the audio to WAV
    3. Run DeepSpeech speech-to-text

2. ``calc``: Calculate the statistics between the YouTube (human, ground-truth) and DeepSpeech (AI, ML transcripts.

    Process: For each processed transcript (those with ``--suffix``) in ``dataset/transcripts``:

    1. Convert the YouTube captions file to a string
    2. Apply pre-processing to the transcripts (to lower case, remove multiple spaces, strip, sentences to list of words, remove empty strings)
    3. Compute the statistics using the `jiwer <https://pypi.org/project/jiwer/>`_ package
    4. Log the stats
    5. When all files are complete then log the average stats

.. note:: This script does not automatically download the transcripts for the YouTube videos. It just transcribes the YouTube videos in ``dataset/videos-dataset.csv`` with DeepSpeech and computes statistics with ground-truth transcripts. This means your ground-truth transcripts can come from a source other than YouTube and this script will still work. To download the transcripts for the videos in ``dataset/videos-dataset.csv`` use :ref:`ss_video_downloader`.

Directions
^^^^^^^^^^

Step 0: Make sure how have some videos in ``dataset/videos-dataset.csv``. The :ref:`ss_youtube_scraper` script can be used to add videos to the dataset.

1. Run ``python 2-video_downloader.py csv --transcript`` to download transcripts (in ".vtt" format) for all the YouTube videos in ``dataset/videos-dataset.csv`` to the ``dataset/transcripts`` folder.
2. Run ``python transcripts_wer.py transcribe`` to transcribe all the videos with ground-truth transcripts using DeepSpeech.
3. Run ``python transcripts_wer.py calc`` to calculate the statistics (including WER) between the DeepSpeech and YouTube transcripts.

Transcripts WER Script Help
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    usage: transcripts_wer.py [-h] [--transcripts_dir TRANSCRIPTS_DIR]
                            [--deepspeech_dir DEEPSPEECH_DIR] [--suffix SUFFIX]
                            [--no_chunk]
                            [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                            {transcribe,calc_wer}

    Word Error Rate (WER) for Transcripts with DeepSpeech

    positional arguments:
    {transcribe,calc_wer}
                            `transcribe` each video and create a transcript using
                            ML models or use `calc_wer` to compute the WER for the
                            created transcripts

    optional arguments:
    -h, --help            show this help message and exit
    --transcripts_dir TRANSCRIPTS_DIR
                            path to the directory containing transcripts
                            downloaded with 2-video_downloader.py
    --deepspeech_dir DEEPSPEECH_DIR
                            path to the directory containing the DeepSpeech models
    --suffix SUFFIX       string added after the video id and before the
                            extension in the transcript output from the ML model
    --no_chunk            Disable audio chunking by voice activity.
    -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --log {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                            Set the logging level (default: 'Info').
