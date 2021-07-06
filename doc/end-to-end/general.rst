.. _e2e_general_info:

E2E General Information
=======================

The end-to-end approach. One command to take a video file and return summarized notes.

Run ``python main.py <path to video>`` to get a notes file. See :ref:`tutorial_general_summarize` for a brief introduction.

Overall Explanation
-------------------

First, frames are extracted once every second. Each frame is classified using the slide classifier (see :ref:`sc_overview`). Next, frames that were classified as ``slide`` are processed by a black border removal algorithm, which is a simple program that crop to the largest rectangle in an image if the edge pixel values of the image are all close to zero. Thus, screen-captured slide frames that have black bars on the sides from a presentation created with a 4:3 aspect ratio but recorded at 16:9 can be interpreted correctly.

Frames that were classified as ``presenter_slide`` are perspective cropped through feature matching and contour/hough lines algorithms. This process removes duplicate slides and crops images from the ``presenter_slide`` class to only contain the side. However, to clean up any duplicates that may remain and to find the best representation of each slide, the slide frames are clustered using our custom ``segment`` clustering algorithm.

At this point, the process has identified the set of unique slides presented in the lecture video. The next step of the pipeline is to process these slides by performing an SSA, which is an algorithm that extracts formatted text from each slide. After that, figures are extracted from the slides and attached to the SSA (see :ref:`slide_structure_analysis`) for each slide. A figure is an image, chart, table, or diagram. After the system has extracted the visual content, it begins processing the audio. The audio is transcribed automatically using the Vosk small 36MB model.

After the audio is transcribed, the system has a textual representation of both visual and auditory data, which need to be combined and summarized to create the final output. If the user desires notes then the SSA will be used for formatting, otherwise, there are tens of different ways of combining and summarizing the audio and slide transcripts, which are discussed in :ref:`e2e_summarization_approaches`.

Script Descriptions
-------------------

These descriptions are short and concise. For more information about some of the larger, more complicated files visit their respective pages on the left.

* **border_removal:** The black border removal algorithm is a simple instruction set that finds the largest rectangle in the image if the edge pixel values of the image are all :math:`<\gamma` and then crops to that rectangle.
* **cluster:** Provides :class:`lecture2notes.end_to_end.cluster.ClusterFilesystem` class, which clusters images from a directory and saves them to disk in folders corresponding to each centroid.
* **corner_crop_transform:** Provides functions to detect the bounding box of a slide in a frame and automatically crop to that bounding box. The :meth:`lecture2notes.end_to_end.corner_crop_transform.all_in_folder` method is used by the main script. See :ref:`corner_crop_transform` for more information. This is one of the two components used to remove duplicate slides and crop ``presenter_slide`` images to only contain the slide. You can learn more about the overall perspective cropping process at :ref:`perspective_cropping`.
* **figure_detection:**: The figure extraction algorithm identifies and saves images, charts, tables, and diagrams from slide frames so that they can be shown in the final summary. See :ref:`figure_detection` for more information.
* **frames_extractor:** Provides :meth:`lecture2notes.end_to_end.frames_extractor.extract_frames`, which extracts frames from ``input_video_path`` at quality level ``quality`` (best quality is 2) every ``extract_every_x_seconds`` seconds and saves them to ``output_path``.
* **helpers:** A small file of helper functions to reduce duplicate code. See :ref:`e2e_api_helpers`.
* **imghash:** Provides functions to detect near duplicate images using image hashing methods from the ``imagehash`` library. :meth:`lecture2notes.end_to_end.imghash.sort_by_duplicates` will create lists of similar images and :meth:`lecture2notes.end_to_end.imghash.remove_duplicates` will remove those duplicates and keep the last file (when sorted alphanumerically descending)
* **main:** The master file that brings all of the components in this directory together by calling the functions provided by the components. Implements a ``skip_to`` variable that can be set to skip to a certain step of the process. This is useful if a pervious step completed but the overall process failed. The ``--help`` is :ref:`located below <e2e_general_main_script_help>`.
* **ocr:** OCR processing uses the `pytesseract <https://pypi.org/project/pytesseract/>`_ (`GitHub <https://github.com/madmaze/pytesseract>`_) package. "Python-tesseract is an optical character recognition (OCR) tool for python. That is, it will recognize and 'read' the text embedded in images. Python-tesseract is a wrapper for `Google’s Tesseract-OCR Engine <https://github.com/tesseract-ocr/tesseract>`_." `This page <https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html>`_ has good information to improve results from tesseract. See :meth:`ocr.all_in_folder` and :meth:`lecture2notes.end_to_end.ocr.write_to_file`.
* **segment_cluster:** :class:`~lecture2notes.end_to_end.segment_cluster.SegmentCluster` iterates through frames in order and splits based on large visual differences. These differences are measured by the cosine difference between the feature vectors (2nd to last layer or right before the softmax) outputted by the slide classifier. This class behaves similarly to :class:`lecture2notes.end_to_end.cluster.ClusterFilesystem` in that it also provides :meth:`~lecture2notes.end_to_end.segment_cluster.SegmentCluster.extract_and_add_features` and :meth:`~lecture2notes.end_to_end.segment_cluster.SegmentCluster.transfer_to_filesystem`.
* **sift_matcher:**: One of the components used to remove duplicate slides and crop ``presenter_slide`` images to only contain the slide. You can learn more about the ``sift_matcher`` at :ref:`feature_matching` and the overall perspective cropping process at :ref:`perspective_cropping`.
* **slide_classifier:** Provides :meth:`lecture2notes.end_to_end.slide_classifier.classify_frames` which automatically sorts images (the extracted frames) using the slide-classifier model. The inference script in ``models/slide_classifier`` is used.
* **spell_check:** Contains the :class:`~lecture2notes.end_to_end.spell_check.SpellChecker` class, which can spell check a string with :meth:`~lecture2notes.end_to_end.spell_check.SpellChecker.check` or a list of strings with :meth:`~lecture2notes.end_to_end.spell_check.SpellChecker.check_all`. With both functions, the best correction is returned.
* **summarization_approaches:** Many summarization models and algorithms for use with ``end_to_end/main.py``. The :meth:`lecture2notes.end_to_end.summarization_approaches.cluster` is probably the most interesting method from this file.
* **transcript_downloader:** Provides the :class:`lecture2notes.end_to_end.transcript_downloader.TranscriptDownloader` class, which downloads transcripts from YouTube using the YouTube API or ``youtube-dl``. ``youtube-dl`` is the recommended method since it does not require an API key and is significantly more reliable than the YouTube API.
* **youtube_api:** Function to use YouTube API with key or ``client_secret.json``. See :meth:`youtube_api.init_youtube`.

.. _e2e_general_main_script_help:

Main Script Help
----------------

Output of ``python -m lecture2notes.end_to_end.main --help``:

.. code-block::

    usage: main.py [-h] [-s N] [-d PATH] [-id] [--custom_id CUSTOM_ID] [-rm]
    [--extract_frames_quality EXTRACT_FRAMES_QUALITY]
    [--extract_every_x_seconds EXTRACT_EVERY_X_SECONDS]
    [--slide_classifier_model_path SLIDE_CLASSIFIER_MODEL_PATH]
    [--east_path EAST_PATH] [-c {silence,speech,none}] [-rd]
    [-cm {normal,segment}]
    [-ca {only_asr,concat,full_sents,keyword_based}]
    [-sm {none,full_sents} [{none,full_sents} ...]]
    [-sx {none,cluster,lsa,luhn,lex_rank,text_rank,edmundson,random}]
    [-sa {none,presumm,sshleifer/distilbart-cnn-12-6,patrickvonplaten/bert2bert_cnn_daily_mail,facebook/bart-large-cnn,allenai/led-large-16384-arxiv,patrickvonplaten/led-large-16384-pubmed,google/pegasus-billsum,google/pegasus-cnn_dailymail,google/pegasus-pubmed,google/pegasus-arxiv,google/pegasus-wikihow,google/pegasus-big_patent}]
    [-ss {structured_joined,none}]
    [--structured_joined_summarization_method {none,abstractive,extractive}]
    [--structured_joined_abs_summarizer {presumm,sshleifer/distilbart-cnn-12-6,patrickvonplaten/bert2bert_cnn_daily_mail,facebook/bart-large-cnn,allenai/led-large-16384-arxiv,patrickvonplaten/led-large-16384-pubmed,google/pegasus-billsum,google/pegasus-cnn_dailymail,google/pegasus-pubmed,google/pegasus-arxiv,google/pegasus-wikihow,google/pegasus-big_patent}]
    [--structured_joined_ext_summarizer {lsa,luhn,lex_rank,text_rank,edmundson,random}]
    [-tm {sphinx,google,youtube,deepspeech,vosk,wav2vec}]
    [--transcribe_segment_sentences]
    [--custom_transcript_check CUSTOM_TRANSCRIPT_CHECK]
    [-sc {ocr,transcript} [{ocr,transcript} ...]] [--video_id ID]
    [--transcribe_model_dir DIR] [--abs_hf_api]
    [--abs_hf_api_overall] [--tensorboard PATH]
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
    --custom_id CUSTOM_ID
            same as `--auto_id` but will create a subdirectory
            using this value instead of a random id
    -rm, --remove         remove `process_dir` once conversion is complete
    --extract_frames_quality EXTRACT_FRAMES_QUALITY
            ffmpeg quality of extracted frames
    --extract_every_x_seconds EXTRACT_EVERY_X_SECONDS
            how many seconds between extracted frames
    --slide_classifier_model_path SLIDE_CLASSIFIER_MODEL_PATH
            path to the slide classification model checkpoint
    --east_path EAST_PATH
            path to the EAST text detector model
    -c {silence,speech,none}, --chunk {silence,speech,none}
            split the audio into small chunks on `silence` using
            PyDub or voice activity `speech` using py-webrtcvad.
            set to 'none' to disable. Recommend 'speech' for
            DeepSpeech and 'none' for Vosk. (default: 'none').
    -rd, --remove_duplicates
            remove duplicate slides before perspective cropping
            and before clustering (helpful when `--cluster_method`
            is `segment`)
    -cm {normal,segment}, --cluster_method {normal,segment}
            which clustering method to use. `normal` uses a
            clustering algorithm from scikit-learn and `segment`
            uses the special method that iterates through frames
            in order and splits based on large visual differences
    -ca {only_asr,concat,full_sents,keyword_based}, --combination_algo {only_asr,concat,full_sents,keyword_based}
            which combination algorithm to use. more information
            in documentation.
    -sm {none,full_sents} [{none,full_sents} ...], --summarization_mods {none,full_sents} [{none,full_sents} ...]
            modifications to perform during summarization process.
            each modification is run between the combination and
            extractive stages. more information in documentation.
    -sx {none,cluster,lsa,luhn,lex_rank,text_rank,edmundson,random}, --summarization_ext {none,cluster,lsa,luhn,lex_rank,text_rank,edmundson,random}
            which extractive summarization approach to use. more
            information in documentation.
    -sa {none,presumm,sshleifer/distilbart-cnn-12-6,patrickvonplaten/bert2bert_cnn_daily_mail,facebook/bart-large-cnn,allenai/led-large-16384-arxiv,patrickvonplaten/led-large-16384-pubmed,google/pegasus-billsum,google/pegasus-cnn_dailymail,google/pegasus-pubmed,google/pegasus-arxiv,google/pegasus-wikihow,google/pegasus-big_patent}, --summarization_abs {none,presumm,sshleifer/distilbart-cnn-12-6,patrickvonplaten/bert2bert_cnn_daily_mail,facebook/bart-large-cnn,allenai/led-large-16384-arxiv,patrickvonplaten/led-large-16384-pubmed,google/pegasus-billsum,google/pegasus-cnn_dailymail,google/pegasus-pubmed,google/pegasus-arxiv,google/pegasus-wikihow,google/pegasus-big_patent}
            which abstractive summarization approach/model to use.
            more information in documentation.
    -ss {structured_joined,none}, --summarization_structured {structured_joined,none}
            An additional summarization algorithm that creates a
            structured summary with figures, slide content (with
            bolded area), and summarized transcript content from
            the SSA (Slide Structure Analysis) and transcript JSON
            data.
    --structured_joined_summarization_method {none,abstractive,extractive}
            The summarization method to use during
            `structured_joined` summarization.
    --structured_joined_abs_summarizer {presumm,sshleifer/distilbart-cnn-12-6,patrickvonplaten/bert2bert_cnn_daily_mail,facebook/bart-large-cnn,allenai/led-large-16384-arxiv,patrickvonplaten/led-large-16384-pubmed,google/pegasus-billsum,google/pegasus-cnn_dailymail,google/pegasus-pubmed,google/pegasus-arxiv,google/pegasus-wikihow,google/pegasus-big_patent}
            The abstractive summarizer to use during
            `structured_joined` summarization (to create summaries
            of each slide) if
            `structured_joined_summarization_method` is
            'abstractive'.
    --structured_joined_ext_summarizer {lsa,luhn,lex_rank,text_rank,edmundson,random}
            The extractive summarizer to use during
            `structured_joined` summarization (to create summaries
            of each slide) if
            `--structured_joined_summarization_method` is
            'extractive'.
    -tm {sphinx,google,youtube,deepspeech,vosk,wav2vec}, --transcription_method {sphinx,google,youtube,deepspeech,vosk,wav2vec}
            specify the program that should be used for
            transcription. CMU Sphinx: use pocketsphinx Google
            Speech Recognition: probably will require chunking
            (online, free, max 1 minute chunks) YouTube: download
            a video transcript from YouTube based on `--video_id`
            DeepSpeech: Use the deepspeech library (fast with good
            GPU) Vosk: Use the vosk library (extremely small low-
            resource model with great accuracy, this is the
            default) Wav2Vec: State-of-the-art speech-to-text
            model through the `huggingface/transformers` library.
    --transcribe_segment_sentences
            Disable DeepSegment automatic sentence boundary
            detection. Specifying this option will output
            transcripts without punctuation.
    --custom_transcript_check CUSTOM_TRANSCRIPT_CHECK
            Check if a transcript file (follwed by an extension of
            vtt, srt, or sbv) with the specified name is in the
            processing folder and use it instead of running
            speech-to-text.
    -sc {ocr,transcript} [{ocr,transcript} ...], --spell_check {ocr,transcript} [{ocr,transcript} ...]
            option to perform spell checking on the ocr results of
            the slides or the voice transcript or both
    --video_id ID         id of youtube video to get subtitles from. set
            `--transcription_method` to `youtube` for this
            argument to take effect.
    --transcribe_model_dir DIR
            path containing the model files for Vosk/DeepSpeech if
            `--transcription_method` is set to one of those
            models. See the documentation for details.
    --abs_hf_api          use the huggingface inference API for abstractive
            summarization tasks
    --abs_hf_api_overall  use the huggingface inference API for final overall
            abstractive summarization task
    --tensorboard PATH    Path to tensorboard logdir. Tensorboard not used if
            not set. Tensorboard only used to visualize cluster
            primarily for debugging.
    -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --log {DEBUG,INFO,WARNING,ERROR,CRITICAL}
            Set the logging level (default: 'Info').
