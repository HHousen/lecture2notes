.. _ss_home:

Scraper Scripts
===============

All scripts that are needed to obtain and manipulate the data. Located in ``dataset/scraper-scripts``.

Note: The number before the name of each script corresponds to the order the scripts are normally used in. Some scripts may have the same number because they do different tasks that take the same spot in the data processing process. For instance, there may be one script to work with slide presentations (PDFs) and another to work with videos that occupy the same position (for instance :ref:`ss_slides_downloader` and :ref:`ss_video_downloader`)

.. _ss_website_scraper:

1. Website Scraper
------------------

Takes a video page link, video download link, and video published date and then adds that information to ``dataset/videos-dataset.csv``.

* Command:

    .. code-block:: bash

        python 1-website_scraper.py <date> <page_link> <video_download_link> <description (optional)>

    * ``<date>`` is the date the lecture was published
    * ``<page_link>`` is the link to the webpage where the video can be found
    * ``<video_download_link>`` is the direct link to the video
    * ``<description (optional)>`` is an optional description that gets saved with the rest of the information (currently not used internally)

* Example:

    .. code-block:: bash

        python 1-website_scraper.py 1-1-2010 \
        https://oyc.yale.edu/astronomy/astr-160/update-1 \
        http://openmedia.yale.edu/cgi-bin/open_yale/media_downloader.cgi?file=/courses/spring07/astr160/mov/astr160_update01_070212.mov

.. _ss_youtube_scraper:

1. YouTube Scraper
------------------

Takes a video id or channel id from YouTube, extracts important information using the YouTube Data API, and then adds that information to ``dataset/videos-dataset.csv``.

* Output of ``python 1-youtube_scraper.py --help``:

    .. code-block:: bash

        usage: 1-youtube_scraper.py [-h] [-n N] [-t] [--transcript-use-yt-api] [-l N]
                            [-f PATH] [-o SEARCH_ORDER] [-p PARAMS]
                            {video,channel,transcript} STR

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
                                `channel`. 50 videos per page.
        -t, --transcript      Download transcript for each video scraped.
        --transcript-use-yt-api
                                Use the YouTube API instead of youtube-dl to download
                                transcripts. `--transcript` must be specified for this
                                option to take effect.
        -l N, --min_length_check N
                                Minimum video length in minutes to be scraped. Only
                                works when `mode` is "channel"
        -f PATH, --file PATH  File to add scraped results to.
        -o SEARCH_ORDER, --search_order SEARCH_ORDER
                                The order to list videos from a channel when `mode` is
                                'channel'. Acceptable values are in the YouTube API
                                Documentation: https://developers.google.com/youtube/v
                                3/docs/search/list
        -p PARAMS, --params PARAMS
                                A string dictionary of parameters to pass to the call
                                to the YouTube API. If mode=video then the
                                `videos.list` api is used. If mode=channel then the
                                `search.list` api is used.

* Examples
    * Add a single lecture video to the dataset:
        .. code-block:: bash

            python 1-youtube_scraper.py video 63hAHbkzJG4
    * Get the transcript for a video file:
        .. code-block:: bash

            python 1-youtube_scraper.py transcript 63hAHbkzJG4
    * Add a video to the ``dataset/videos-dataset.csv`` and get the transcript:
        .. code-block:: bash

            python 1-youtube_scraper.py video 63hAHbkzJG4 --transcript
    * Scrape the 50 latest videos from a channel:
        .. code-block:: bash

            python 1-youtube_scraper.py channel UCEBb1b_L6zDS3xTUrIALZOw --num_pages 1
    * Scrape the 50 most viewed videos from a channel:
        .. code-block:: bash

            python 1-youtube_scraper.py channel UCEBb1b_L6zDS3xTUrIALZOw --num_pages 1 --search_order viewCount
    * Scrape the 50 latest videos from a channel that were published before 2020:
        .. code-block:: bash

            python 1-youtube_scraper.py channel UCEBb1b_L6zDS3xTUrIALZOw --num_pages 1 --params '{"publishedBefore": "2020-01-01T00:00:00Z"}'
    * Scrape the 100 latest videos from a channel longer than 20 minutes:
        .. code-block:: bash

            python 1-youtube_scraper.py channel UCEBb1b_L6zDS3xTUrIALZOw --num_pages 2 --min_length_check 20
    * **Mass Download 1** (to be used with :ref:`ss_mass_data_collector`):
        .. code-block:: bash

            python 1-youtube_scraper.py channel UCEBb1b_L6zDS3xTUrIALZOw --num_pages 2 --min_length_check 20 -f ../mass-download-list.csv
    * **Mass Download 2** (specify certain dates and times):
        .. code-block:: bash

            python 1-youtube_scraper.py channel UCEBb1b_L6zDS3xTUrIALZOw --num_pages 2 --min_length_check 20 -f ../mass-download-list.csv --params '{"publishedBefore": "2015-01-01T00:00:00Z", "publishedAfter": "2014-01-01T00:00:00Z"}'

.. _ss_mass_data_collector:

2. Mass Data Collector
----------------------

This script provides a method to collect massive amounts of new data for the slide classifier. These new lecture videos are selected based on what the model struggles with (where its certainty is lowest). This means the collected videos train the model the fastest while exposing it to the most unique situations. However, this method will ignore videos that the model is very confident with but is actually incorrect. These videos are the most beneficial but must be manually found.

The *Mass Data Collector* does the following for each video in ``dataset/mass-download-list.csv``:
    1. Download the video to ``dataset/mass-download-temp/[video_id]``
    2. Extracts frames
    3. Classifies the frames to obtain certainties and the percent incorrect (where certainty is below a threshold)
    4. Adds ``video_id``, ``average_certainty``, ``num_incorrect``, ``percent_incorrect``, and ``certainties`` to ``dataset/mass-download-results.csv``
    5. Deletes video folder (``dataset/mass-download-temp/[video_id]``)

The ``--top-k`` (or ``-k``) argument can be specified to the script add the top ``k`` most uncertain videos to the ``dataset/videos-dataset.csv``. This must be ran after the ``dataset/mass-download-results.csv`` file has been populated.

.. warning::
    This script will use a lot of bandwidth/data. For instance, the below commands will download 100 videos from YouTube. If each video is 100MB (which is likely on the low end) then this will download at least 10GB of data.

Examples:

1. Low Disk Space Usage, High Bandwidth, Duplicate Calculations, Large Dataset Filesize
    **Recommended** if you want to build the dataset at full 1080p resolution so that it can be used with a plethora of model architectures. This was how the official dataset was compiled.

    The below commands do the following:

    1. Scrape the `MIT OpenCourseWare <https://www.youtube.com/channel/UCEBb1b_L6zDS3xTUrIALZOw>`_ YouTube channel for the latest 100 videos that are longer than 20 minutes and save the data to ``../mass-download-list.csv``

        * Optionally, only find videos in a date range. To do this you need to specify the ``--params`` argument like so: ``--params '{"publishedBefore": "2014-07-01T00:00:00Z", "publishedAfter": "2014-01-01T00:00:00Z"}'``. The full list of available parameters can be found in the `YouTube API Documentation for search.list <https://developers.google.com/youtube/v3/docs/search/list>`_ if  mode is ``channel`` and `YouTube API Documentation for videos.list <https://developers.google.com/youtube/v3/docs/videos/list>`_ if mode is ``video``.

    2. Run the *Mass Data Collector* to download each video at 480p and determine how certain the model is with its predictions on that video.
    3. Take the top 20 most uncertain videos and add them to the ``dataset/videos-dataset.csv``.
    4. Download the newly added 20 videos at 480p
    5. Extract frames from the new videos
    6. Sort the frames from top 20 most uncertain videos
    7. Now it is time for you to check the model's predictions, fix them, and then train a better model on the new data.

    .. code-block:: bash

        python 1-youtube_scraper.py channel UCEBb1b_L6zDS3xTUrIALZOw --num_pages 2 --min_length_check 20 -f ../mass-download-list.csv
        python 2-mass_data_collector.py --resolution 480
        python 2-mass_data_collector.py -k 20
        python 2-video_downloader.py csv
        python 3-frame_extractor.py auto
        python 4-auto_sort.py

2. High Disk Space Usage, Higher Bandwidth, *No* Duplicate Calculations, Large Dataset Filesize
    **Recommended** if you want to build the dataset at full 1080p resolution but do not want to "waste" compute resources on duplicate calculations.

    Specifying the ``--no_remove`` argument to ``2-mass_data_collector.py`` will make the script keep the processed videos instead of removing them. This means the videos can be copied to the ``dataset/videos`` folder, manually inspected and fixed, and then :ref:`ss_compile_data` can be used to copy them to the ``dataset/classifier-data`` folder.

    It is recommended to not set the ``--resolution`` if using this method because some of the downloaded videos will eventually be added to the dataset. **The dataset is compiled at maximum resolution so that different models can be used that accept different resolutions.**

3. Lower Disk Space Usage, Low Bandwidth, Duplicate Calculations, Small Dataset Filesize
    **Recommended** if you want to build the dataset for a specific model architecture and if you want the dataset to take up a relatively small amount of disk space.

    If you want to train a ``resnet34``, for example, which expects 224x224 input images, then you can set the resolution to 240p when downloading videos since the frames will be scaled before being used for training anyway. However, if you ever want to train a model that expects larger input images, you will have to download and reprocess the entire dataset.

    The modified commands look like this:

    .. code-block:: bash

        python 1-youtube_scraper.py channel UCEBb1b_L6zDS3xTUrIALZOw --num_pages 2 --min_length_check 20 -f ../mass-download-list.csv
        python 2-mass_data_collector.py --resolution 240
        python 2-mass_data_collector.py -k 20
        python 2-video_downloader.py csv --resolution 240
        python 3-frame_extractor.py auto
        python 4-auto_sort.py

    Notice that the resolution was changed to 240 for the second command and the resolution option was added to the fourth command.

    This option can be modified as described in the second method by adding the ``--no_remove`` argument to ``2-mass_data_collector.py``. This will increase disk usage but will prevent duplicate calculations and decrease overall bandwidth since videos will not have to be redownloaded.

Mass Dataset Collector Script Help
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Output of ``python 2-mass_data_collector.py --help``:

    .. code-block:: bash

        usage: 2-mass_data_collector.py [-h] [-k K] [-nr] [-r RESOLUTION] [-p]

        Mass Data Collector

        optional arguments:
        -h, --help            show this help message and exit
        -k K, --top_k K       Add the top `k` most uncertain videos to the videos-
                                dataset.
        -nr, --no_remove      Don't remove the videos after they have been
                                processed. This makes it faster to manually look
                                through the most uncertain videos since they don't
                                have to be redownloaded, but it will use more disk
                                space.
        -r RESOLUTION, --resolution RESOLUTION
                                The resolution of the videos to download. Default is
                                maximum resolution.
        -p, --pause           Pause after each video has been processed but before
                                deletion.

.. _ss_slides_downloader:

2. Slides Downloader
--------------------

Takes a link to a pdf slideshow and downloads it to ``dataset/slides/pdfs`` or downloads every entry in ``dataset/slides-dataset.csv`` (*csv* option).

* Command: `python slides_downloader.py <csv/your_url>`
* Examples:
    * If *csv*: ``python 2-slides_downloader.py csv``
    * If *your_url*: ``python 2-slides_downloader.py https://bit.ly/3dYtUPM``
* Required Software: ``wget``

.. _ss_video_downloader:

2. Video Downloader
-------------------

Uses ``youtube-dl`` (for ``youtube`` videos) and ``wget`` (for ``website`` videos) to download either a youtube video by id or every video that has not been download in ``dataset/videos-dataset.csv``.

This script can also download the transcripts from YouTube using ``youtube-dl`` for each video in ``dataset/videos-dataset.csv`` with the ``--transcript`` argument..

* Command: `python 2-video_downloader.py <csv/youtube --video_id your_youtube_video_id>`
* Examples:
    * If *csv*: ``python 2-video_downloader.py csv``
    * If *your_youtube_video_id*: ``python 2-video_downloader.py youtube --video_id 1Qws70XGSq4``
    * Download all transcripts: ``python 2-video_downloader.py csv --transcript`` (will not download videos or change ``dataset/videos-dataset.csv``)
* Required Software: ``youtube-dl`` (`YT-DL Website <https://ytdl-org.github.io/youtube-dl/index.html>`_/`YT-DL Github <https://github.com/ytdl-org/youtube-dl>`_), ``wget``

Video Downloader Script Help
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Output of ``python 2-video_downloader.py --help``:

.. code-block:: bash

    usage: 2-video_downloader.py [-h] [--video_id VIDEO_ID] [--transcript]
                                [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                                {csv,youtube}

    Video Downloader

    positional arguments:
    {csv,youtube}         `csv`: Download all videos that have not been marked
                            as downloaded from the `videos-dataset.csv`.
                            `youtube`: download the specified video from YouTube
                            with id ``--video_id`.

    optional arguments:
    -h, --help            show this help message and exit
    --video_id VIDEO_ID   The YouTube video id to download if `method` is
                            `youtube`.
    --transcript          Download the transcript INSTEAD of the video for each
                            entry in `videos-dataset.csv`. This ignores the
                            `downloaded` column in the CSV and will not download
                            videos.
    -r RESOLUTION, --resolution RESOLUTION
                            The resolution of the videos to download. Default is
                            maximum resolution.
    -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --log {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                            Set the logging level (default: 'Info').

.. _ss_frame_extractor:

3. Frame Extractor
------------------

Extracts either every N frames from a video file (selected by id and must be in `videos` folder) or, in ``auto`` mode, every N frames from every video in the dataset that has been downloaded and has not had its frames extracted already. ``extract_every_x_seconds`` can be set to auto to use the ``get_extract_every_x_seconds()`` function to automatically determine a good number of frames to extract. ``auto`` mode uses this feature and allows for exact reconstruction of the dataset. Extracted frames are saved into ``dataset/videos/[video_id]/frames``.

* Command: ``python 3-frame_extractor.py <video_id/auto> <extract_every_x_seconds/auto> <quality>``
* Examples:
    * If *video_id*: ``python 3-frame_extractor.py VT2o4KCEbes 20 5`` or to automatically extract a good number of frames: ``python 3-frame_extractor.py 63hAHbkzJG4 auto 5``
    * If *auto*:  ``python 3-frame_extractor.py auto``
* Required Software: ``ffmpeg`` (`FFmpeg Website <https://www.ffmpeg.org/>`_/`FFmpeg Github <https://github.com/FFmpeg/FFmpeg>`_)

.. _ss_pdf2image:

3. pdf2image
------------

Takes every page in all pdf files in ``dataset/slides/pdfs``, converts them to png images, and saves them in ``dataset/slides/images/pdf_file_name``.

* Command: ``python 3-pdf2image.py``
* Required Software: ``poppler-utils (pdftoppm)`` (`Man Page <https://linux.die.net/man/1/pdftoppm>`_/`Website <https://poppler.freedesktop.org/>`_)

.. _ss_auto_sort:

4. Auto Sort
------------

Goes through every extracted frame for all videos in the dataset that don’t have sorted frames (based on the presence of the ``sorted_frames`` directory) and classifies them using ``models/slide_classifier``. You need either a trained pytorch model to use this. Creates a list of frames that need to be checked for correctness by humans in ``dataset/to-be-sorted.csv``. This script imports certain files from ``models/slide_classifier`` so the directory structure must not have been changed from installation.

* Command: `python 4-auto_sort.py`

.. _ss_sort_from_file:

4. Sort From File
-----------------

Creates a CSV of the category assigned to each frame of each video in the dataset or organizes extracted frames from a previously created CSV. The purpose of this script is to exactly reconstruct the dataset without downloading the already sorted images.

There are three options:
1. ``make``: make a file mapping of the category to which each frame belongs by reading data from the ``dataset/videos`` directory.
2. ``make_compiled`` performs the same task as ``make`` but reads from the ``dataset/classifier-data`` directory. This is useful if the dataset has been compiled and the ``dataset/videos`` folder has been cleared.
3. ``sort``: sort each file in ``dataset/sort_file_map.csv``, moving the respective frame from ``video_id/frames`` to ``video_id/frames_sorted/category``.

.. note:: This script appends to ``dataset/sort_file_map.csv``. It will not overwrite data.

* Command: ``python 4-sort_from_file.py <make/make_compiled/sort>``

.. _ss_compile_data:

5. Compile Data
---------------

Merges the sorted frames from all the ``videos`` and ``slides`` in the dataset to ``dataset/classifier-data``.

.. note:: This script will not erase any data already stored in the ``dataset/classifier-data`` dataset folder.

* Command: ``python 5-compile_data.py <all/videos/slides>``
* Examples:
    * If *videos*: ``python 5-compile_data.py videos``, processes only sorted frames from ``videos``
    * If *slides*:  ``python 5-compile_data.py slides``, processes images from ``slides``
    * If *all*: ``python 5-compile_data.py all``, processes from both ``videos`` and ``slides``
