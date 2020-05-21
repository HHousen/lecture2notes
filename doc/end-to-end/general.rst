.. _e2e_general_info:

E2E General Information
=======================

The end-to-end approach. One command to take a video file and return summarized notes.

Run ``python main.py <path to video>`` to get a notes file. See :ref:`tutorial_general_summarize` for a brief introduction.

Script Descriptions
-------------------

These descriptions are short and concise. For more information about some of the larger, more complicated files visit their respective pages on the left.

* **cluster:** Provides :class:`cluster.ClusterFilesystem` class, which clusters images from a directory and saves them to disk in folders corresponding to each centroid.
* **corner_crop_transform:** Provides functions to detect the bounding box of a slide in a frame and automatically crop to that bounding box. The :meth:`corner_crop_transform.all_in_folder` method is used by the main script. See :ref:`corner_crop_tranform` for more information.
* **frames_extractor:** Provides :meth:`frames_extractor.extract_frames`, which extracts frames from ``input_video_path`` at quality level ``quality`` (best quality is 2) every ``extract_every_x_seconds`` seconds and saves them to ``output_path``.
* **helpers:** A small file of helper functions to reduce duplicate code. See :ref:`e2e_api_helpers`.
* **imghash:** Provides functions to detect near duplicate images using image hashing methods from the ``imagehash`` library. :meth:`imghash.sort_by_duplicates` will create lists of similar images and :meth:`imghash.remove_duplicates` will remove those duplicates and keep the last file (when sorted alphanumerically descending)
* **main:** The master file that brings all of the components in this directory together by calling the functions provided by the components. Implements a ``skip_to`` variable that can be set to skip to a certain step of the process. This is useful if a pervious step completed but the overall process failed. The ``--help`` is :ref:`located below <e2e_general_main_script_help>`.
* **ocr:** OCR processing uses the `pytesseract <https://pypi.org/project/pytesseract/>`_ (`GitHub <https://github.com/madmaze/pytesseract>`_) package. "Python-tesseract is an optical character recognition (OCR) tool for python. That is, it will recognize and 'read' the text embedded in images. Python-tesseract is a wrapper for `Google’s Tesseract-OCR Engine <https://github.com/tesseract-ocr/tesseract>`_." `This page <https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html>`_ has good information to improve results from tesseract. See :meth:`ocr.all_in_folder` and :meth:`ocr.write_to_file`.
* **segment_cluster:** :class:`~segment_cluster.SegmentCluster` iterates through frames in order and splits based on large visual differences. These differences are measured by the cosine difference between the feature vectors (2nd to last layer or right before the softmax) outputted by the slide classifier. This class behaves similarly to :class:`cluster.ClusterFilesystem` in that it also provides :meth:`~segment_cluster.SegmentCluster.extract_and_add_features` and :meth:`~segment_cluster.SegmentCluster.transfer_to_filesystem`.
* **slide_classifier:** Provides :meth:`slide_classifier.classify_frames` which automatically sorts images (the extracted frames) using the slide-classifier model. The inference script in ``Models/slide-classifier`` is used.
* **spell_check:** Contains the :class:`~spell_check.SpellChecker` class, which can spell check a string with :meth:`~spell_check.SpellChecker.check` or a list of strings with :meth:`~spell_check.SpellChecker.check_all`. With both functions, the best correction is returned.
* **summarization_approaches:** Many summarization models and algorithms for use with ``End-To-End/main.py``. The :meth:`summarization_approaches.cluster` is probably the most interesting method from this file.
* **transcript_downloader:** Provides the :class:`transcript_downloader.TranscriptDownloader` class, which downloads transcripts from YouTube using the YouTube API or ``youtube-dl``. ``youtube-dl`` is the recommended method since it does not require an API key and is significantly more reliable than the YouTube API.
* **youtube_api:** Function to use YouTube API with key or ``client_secret.json``. See :meth:`youtube_api.init_youtube`.

.. _e2e_general_main_script_help:

Main Script Help
----------------

Output of `python main.py --help`:

.. code-block::

    usage: main.py [-h] [-s N] [-d PATH] [-id] [-rm] [-c {silence,speech,none}]
               [-rd] [-cm {normal,segment}]
               [-ca {only_asr,concat,full_sents,keyword_based}]
               [-sm {none,full_sents} [{none,full_sents} ...]]
               [-sx {none,cluster,lsa,luhn,lex_rank,text_rank,edmundson,random}]
               [-sa {none,bart,presumm}]
               [-tm {sphinx,google,youtube,deepspeech}]
               [-sc {ocr,transcript} [{ocr,transcript} ...]] [--video_id ID]
               [--deepspeech_model_dir DIR] [--tensorboard PATH]
               [--bart_checkpoint PATH] [--bart_state_dict_key PATH]
               [--bart_fairseq] [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
               DIR

    End-to-End Conversion of Lecture Videos to Notes using ML

    positional arguments:
    DIR                   path to video

    optional arguments:
    -h, --help            show this help message and exit
    -s N, --skip_to N     set to > 0 to skip specific processing steps
    -d PATH, --process_dir PATH
                            path to the proessing directory (where extracted
                            frames and other files are saved), set to "automatic"
                            to use the video's folder (default: ./)
    -id, --auto_id        automatically create a subdirectory in `process_dir`
                            with a unique id for the video and change
                            `process_dir` to this new directory
    -rm, --remove         remove `process_dir` once conversion is complete
    -c {silence,speech,none}, --chunk {silence,speech,none}
                            split the audio into small chunks on `silence` using
                            PyDub or voice activity `speech` using py-webrtcvad.
                            set to 'none' to disable. (default: 'speech').
    -rd, --remove_duplicates
                            remove duplicate slides before clusterting (helpful
                            when `--cluster_method` is `segment`
    -cm {normal,segment}, --cluster_method {normal,segment}
                            which clustering method to use. `normal` uses a
                            clustering algorithm from scikit-learn and `segment`
                            uses the special method that iterates through frames
                            in order and splits based on large visual differences
    -ca {only_asr,concat,full_sents,keyword_based}, --combination_algo {only_asr,concat,full_sents,keyword_based}
                            which extractive summarization approach to use. more
                            information in documentation.
    -sm {none,full_sents} [{none,full_sents} ...], --summarization_mods {none,full_sents} [{none,full_sents} ...]
                            modifications to perform during summarization process.
                            each modification is run between the combination and
                            extractive stages. more information in documentation.
    -sx {none,cluster,lsa,luhn,lex_rank,text_rank,edmundson,random}, --summarization_ext {none,cluster,lsa,luhn,lex_rank,text_rank,edmundson,random}
                            which extractive summarization approach to use. more
                            information in documentation.
    -sa {none,bart,presumm}, --summarization_abs {none,bart,presumm}
                            which abstractive summarization approach/model to use.
                            more information in documentation.
    -tm {sphinx,google,youtube,deepspeech}, --transcription_method {sphinx,google,youtube,deepspeech}
                            specify the program that should be used for
                            transcription. CMU Sphinx: use pocketsphinx (works
                            offline) Google Speech Recognition: probably will
                            require chunking YouTube: pull a video transcript from
                            YouTube based on `--video_id` DeepSpeech: Use the
                            deepspeech library (works offline with great accuracy)
    -sc {ocr,transcript} [{ocr,transcript} ...], --spell_check {ocr,transcript} [{ocr,transcript} ...]
                            option to perform spell checking on the ocr results of
                            the slides or the voice transcript or both
    --video_id ID         id of youtube video to get subtitles from. set
                            `--transcription_method` to `youtube` for this
                            argument to take effect.
    --deepspeech_model_dir DIR
                            path containing the DeepSpeech model files. See the
                            documentation for details.
    --tensorboard PATH    Path to tensorboard logdir. Tensorboard not used if
                            not set. Tensorboard only used to visualize cluster
                            primarily for debugging.
    --bart_checkpoint PATH
                            [BART Abstractive Summarizer Only] Path to optional
                            checkpoint. Semsim is better model but will use more
                            memory and is an additional 5GB download. (default:
                            none, recommended: semsim)
    --bart_state_dict_key PATH
                            [BART Abstractive Summarizer Only] model state_dict
                            key to load from pickle file specified with
                            --bart_checkpoint (default: "model")
    --bart_fairseq        [BART Abstractive Summarizer Only] Use fairseq model
                            from torch hub instead of huggingface transformers
                            library models. Can not use --bart_checkpoint if this
                            option is supplied.
    -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --log {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                            Set the logging level (default: 'Info').
