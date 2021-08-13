.. _install:

Installation
============

Overview
--------

Installation is made easy due to conda environments. Simply run ``conda env create -f environment.yml`` from the root project directory and conda will create an environment called ``lecture2notes`` with all the required packages from ``environment.yml``.

..note:: Read `the paper <https://haydenhousen.com/media/lecture2notes-paper-v1.pdf>`__ for more in-depth explanations regarding the background, methodology, and results of this project.

Info About Optional Components
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Certain functions in the End-To-End ``transcribe.py`` file require additional downloads. If you are not using the transcribe feature of the End-To-End approach then this notice can safely be ignored. These extra files may not be necessary depending on your configuration. To use the similarity function to compare two transcripts a spacy model is needed, which you can learn more about on the spacy `starter models <https://spacy.io/models/en-starters>`_ and `core models <https://spacy.io/models/en>`_ documentation.

The default transcription method in the End-To-End process is to use ``vosk``. You need to download a ``vosk`` model from the `models page <https://alphacephei.com/vosk/models>`_ (`Google Drive Mirror <https://drive.google.com/drive/folders/10OY4DeD3Lm-hdTFLFDvaDYaL2m-8h-Cr>`__) to use this method or you can specify a different method with the ``--transcription_method`` flag such as ``--transcription_method wav2vec``.

The End-To-End ``figure_detection.py`` contains a function called :meth:`~figure_detection.detect_figures`. This function requires the `EAST (Efficient and Accurate Scene Text Detector) <https://arxiv.org/abs/1704.03155>`_ model by default due to the ``do_text_check`` argument defaulting to ``True``. See the docstring for more information. You can download the model from `Dropbox <https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz?dl=1>`__ (this link was extracted from the `official code <https://github.com/opencv/opencv_extra/blob/65796edadce27ed013e3deeedb3c081ff527e4ec/testdata/dnn/download_models.py#L349>`__) or `Google Drive <https://drive.google.com/uc?id=1ZVn7_g58g4B0QNYNFE6MzRzpirsNTjwe>`__ (my mirror). Then just extract the file by running ``tar -xzvf frozen_east_text_detection.tar.gz``.

.. _quick_install:

Quick-Install (Copy & Paste)
----------------------------

.. code-block:: bash

    git clone https://github.com/HHousen/lecture2notes.git
    cd lecture2notes
    conda env create
    conda activate lecture2notes
    python -m spacy download en_core_web_sm
    gdown "https://drive.google.com/uc?id=1eXwWQujo_0HVffuUx0Fa6KydjW8h4gUb" -O lecture2notes/end_to_end/model_best.ckpt

Extras (Linux Only):

Install extras only after the above commands have been run.

.. code-block:: bash

    sudo apt install curl
    sudo curl -L https://yt-dl.org/downloads/latest/youtube-dl -o /usr/local/bin/youtube-dl
    sudo chmod a+rx /usr/local/bin/youtube-dl
    sudo apt install ffmpeg sox wget poppler-utils

Commands to download a Vosk model (needed for speech-to-text) are available on the :ref:`transcribe_method_vosk` transcription method page.

Step-by-Step Instructions
-------------------------

1. Clone this repository: ``git clone https://github.com/HHousen/lecture2notes.git``.
2. Change to project directory: ``cd lecture2notes``.
3. Run installation command: ``conda env create``.
4. Activate newly created conda environment: ``conda activate lecture2notes``.
5. Run `gdown "https://drive.google.com/uc?id=1eXwWQujo_0HVffuUx0Fa6KydjW8h4gUb" -O lecture2notes/end_to_end/model_best.ckpt` from the project root to download the :ref:`slide classification model <sc_overview>` and put it in the default expected location.
6. **Other Binary Packages:** Install ``ffmpeg``, ``sox``, ``wget``, and ``poppler-utils`` with ``sudo apt install ffmpeg sox wget poppler-utils`` if on linux. Otherwise, navigate to the `sox homepage <http://sox.sourceforge.net/>`_ to download ``sox``, the `youtube-dl homepage <https://ytdl-org.github.io/youtube-dl/index.html>`_ (`GitHub <https://github.com/ytdl-org/youtube-dl>`_) to download ``youtube-dl``, and follow the directions in this `StackOverflow answer <https://stackoverflow.com/a/53960829>`_ (Windows) to install ``poppler-utils`` for your platform. ``ffmpeg`` is needed for frame extraction in ``Dataset`` and ``End-To-End``. ``sox`` is needed for automatic audio conversion during the transcription phase of ``End-To-End``. [#f1]_ ``wget`` is used to download videos that are not on youtube as part of the ``video_downloader`` scraper script in ``Dataset``.
7. **End-To-End Process Requirements (Optional)**
    1. **Spacy:** Download the small spacy model by running ``python -m spacy download en_core_web_sm`` in the project root. This is required to use certain summarization and similarity features (as discussed above). A spacy model is also required when using spacy as a feature extractor in ``end_to_end/summarization_approaches.py``. [#f2]_
    2. **DeepSpeech/Vosk**: Download the ``DeepSpeech`` model (the ``.pbmm`` acoustic model and the scorer) from the `releases page <https://github.com/mozilla/DeepSpeech/releases>`_. To reduce complexity save them to ``deepspeech-models`` in the project root. [#f3]_ **Alternatively, it is recommended** to download the small vosk model using the commands on the :ref:`transcribe_method_vosk` transcription method page.
    3. **EAST**: Download the ``EAST`` model from `Dropbox <https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz?dl=1>`__ or by running ``gdown https://drive.google.com/uc?id=1ZVn7_g58g4B0QNYNFE6MzRzpirsNTjwe``. Extract it to the ``End-To-End`` directory by running ``tar -xzvf frozen_east_text_detection.tar.gz -C end_to_end/``
8. **Dataset Collection Requirements (Optional)** YouTube API
    1. Run ``cp .env.example .env`` to create a copy of the example ``.env`` file.
    2. Add your YouTube API key to your ``.env`` file.
    3. You can now use the scraper scripts to scrape YouTube and create the dataset needed to train the slide classifier.
9. **Transcript Download w/YouTube API (Not Recommended)** If you want to download video transcripts with the YouTube API [#f4]_, place your ``client_secret.json`` in the ``dataset/scraper-scripts`` folder (if you want to download transcripts with the ``scraper-scripts``) or in ``End-To-End`` (if you want to download transcripts in the entire end-to-end process that converts a lecture video to notes).

.. rubric:: Footnotes

.. [#f1] If your audio is 16000Hz, 1 channel, and ``.wav`` format, then ``sox`` is not needed.
.. [#f2] The default is *not* to use spacy for feature extraction but the large model (which can be downloaded with ``python -m spacy download en_core_web_lg``) *is* the default if spacy is manually chosen. So make sure to download the large model if you want to use spacy for feature extraction.
.. [#f3] Folder name and location do not matter. Just make sure the scorer and model are in the same directory. The scripts will automatically detect each when given the path to the folder containing them.
.. [#f4] The default is to use ``youtube-dl`` which needs no API key.
