E2E Transcribe (Speech-To-Text)
===============================

API Documentation: :ref:`e2e_api_transcribe`

The ``transcribe`` module contains the features necessary to convert a wave file to text.

There are 4 methods, which rely on 3 libraries, of transcribing audio implemented in this module. Additionally, 2 chunking algorithms are supported. 

Chunking
--------

Chunking increases speed of voice-to-text by reducing the amount of audio without speech.

* **Voice Activity:** Uses the WebRTC Voice Activity Detector (VAD) which is "reportedly one of the best VADs available, being fast, modern, and free." The python bindings to the WebRTC VAD API are provided by `wiseman/py-webrtcvad <https://github.com/wiseman/py-webrtcvad>`_. This algorithm is implemented in the :meth:`~transcribe_main.chunk_by_speech` function. It produces ``segments`` that can be transcribed with :meth:`~transcribe_main.process_segments` using DeepSpeech.
* **Noise Activity:** Detects and removes segments of audio that are significantly below the average loudness of the file. This algorithm is implemented in :meth:`~transcribe_main.chunk_by_silence` and it writes ``chunks`` (wave files that are parts of the original file that contain noise) to disk. :meth:`~transcribe_main.process_chunks` runs transcription on every file in a directory. It can be used in conjunction with :meth:`~transcribe_main.chunk_by_silence` to transcribe all the ``chunk`` and return a merged transcript. If the ``method`` is ``deepspeech`` :meth:`~transcribe_main.transcribe_audio_deepspeech` will be used, otherwise :meth:`~transcribe_main.transcribe_audio` will be used and method will be passed along.

.. note:: Chunking is necessary for the ``google`` method (see :ref:`transcribe_methods`) for long audio files since Google Speech Recognition will time out if the filesize is too large.

.. _transcribe_methods:

Transcribing Methods
--------------------

The recommended method is DeepSpeech since it works offline (``google`` does not), has good accuracy (better than ``sphinx``, accuracy of ``google`` unknown), is actively maintained (``sphinx`` and ``google`` are dated), is completely open source (``google`` is mysterious), and is the most reliable (``google`` frequently times out and ``sphinx`` is difficult to work with (including installation and actual processing of files).

Additionally, it is recommended to combine DeepSpeech with ``--chunk`` set to ``speech`` (``speech`` is better than ``silence``) since this will give a progress indicator and probably increase speed. If DeepSpeech is run without chunking then the process will seem to hang, especially on longer audio files. Simple tests indicate that chunking can improve speed by up to 60%, but also can cause a 12% increase in WER (12% worse).

1. **YouTube:** (:meth:`transcribe.transcribe_main.get_youtube_transcript`) This method only works if the lecture to be summarized is a YouTube video that contains manually added captions. 

    .. figure:: ../_static/captions_vs_no-captions.jpg
        :alt: Image showing two YouTube video thumbnails, one with the CC icon and one without.

        You can tell if a video contains manual captions if it contains the CC icon as shown above.

    This method downloads the transcript for the specified language directly from YouTube using either the YouTube API (:meth:`~transcript_downloader.TranscriptDownloader.get_transcript_api`) or ``youtube-dl`` (:meth:`~transcript_downloader.TranscriptDownloader.get_transcript_ytdl`). Both methods are part of the :class:`~transcript_downloader.TranscriptDownloader` class.
    
    The :meth:`~transcript_downloader.TranscriptDownloader.download` function provides easy access to both of these download options.
    
    .. note:: The YouTube API requires an API key. You can find more information about how to obtain a key for free from `Google Developers <https://developers.google.com/youtube/registering_an_application>`_.

    .. important:: Using ``youtube-dl`` is recommended over the YouTube API because it does not require an API key and is significantly more reliable than the YouTube API. 
    

2. **General:** Sphinx and Google

    The ``sphinx`` and ``google`` methods use the `SpeechRecognition library <https://pypi.org/project/SpeechRecognition/>`_ to access ``pockersphinx-python`` and Google Speech Recognition, respectively. These methods are grouped together in the :meth:`~transcribe_main.transcribe_audio` function because the SpeechRecognition library simplifies the differences to one line. The ``method`` argument allows the switching between both methods. 

    .. note:: The ``google`` method uses "Google Speech Recognition" (free) and not the `Google Cloud Speech API <https://cloud.google.com/speech/>`_ (paid). It is my understanding that "Google Speech Recognition" is deprecated and could disappear anytime.
    

3. **DeepSpeech:**

    The ``deepspeech`` method uses the `Mozilla DeepSpeech <https://github.com/mozilla/DeepSpeech>`_ library, which achieves very good accuracy on the `LibriSpeech clean test corpus <https://www.openslr.org/12>`_ (the current model accuracy can be found on the `latest release page <https://github.com/mozilla/DeepSpeech/releases/latest>`_. 
    
    The DeepSpeech architecture was created by *Baidu* in 2014. Project DeepSpeech was created by *Mozilla* (the creators of the popular Firefox web browser) to provide the open source community with an updated Speech-To-Text engine.
    
    In order to use this method in the ``end_to_end/main.py`` script you  the latest DeepSpeech model needs to be downloaded from the `releases page <https://github.com/mozilla/DeepSpeech/releases>`_. Mozilla provides code to download and extract the model on the `project's documentation <https://deepspeech.readthedocs.io/en/latest/USING.html#getting-the-pre-trained-model>`_. You can rename these files as long as the extensions remain the same. When using the ``end_to_end/main.py`` script you only have to specify the directory containing both files (the directory name is not important but `deepspeech-models` is descriptive). See :ref:`install` for more details about downloading the deepspeech models.
    
    Example Folder Structure:
    .. code-block:: bash
    
        deepspeech-models/
        ├── deepspeech-0.7.1-models.pbmm
        ├── deepspeech-0.7.1-models.scorer

.. note:: There are better public models than DeepSpeech. However, they exist in research-focused libraries that make inference difficult. Go to :ref:`transcribe_other_options` for more info.

Script Descriptions
-------------------

* **transcribe_main**: Implements transcription using four different methods from 3 libraries and other miscellaneous functions related to audio transcription, including audio reading, writing, extraction, and conversion.
* **webrtcvad_utils**: Implements functions to filter out non-voiced sections from audio files. The primary function is :meth:`~webrtcvad_utils.vad_segment_generator`, which accepts an audio path and returns segments of audio with voice.
* **mic_vad_streaming**: Streams from microphone to DeepSpeech, using Voice Activity Detection (VAD) provided by ``webrtcvad``. This is essentially the `example file <https://github.com/mozilla/DeepSpeech-examples/blob/r0.7/mic_vad_streaming/mic_vad_streaming.py>`_ from `mozilla/DeepSpeech-examples <https://github.com/mozilla/DeepSpeech-examples>`_.
    * To select the correct input device, the code below can be used. It will print a list of devices and associated parameters as detected by ``pyaudio``.
    
    .. code-block:: bash
    
        import pyaudio
        p = pyaudio.PyAudio()
        for i in range(p.get_device_count()):
            print(p.get_device_info_by_index(i))
    
    * Output of ``python mic_vad_streaming.py --help``

    .. code-block:: bash
    
        usage: mic_vad_streaming.py [-h] [-v VAD_AGGRESSIVENESS] [--nospinner]
                                    [-w SAVEWAV] [-f FILE] -m MODEL [-s SCORER]
                                    [-d DEVICE] [-r RATE]

        Stream from microphone to DeepSpeech using VAD

        optional arguments:
        -h, --help            show this help message and exit
        -v VAD_AGGRESSIVENESS, --vad_aggressiveness VAD_AGGRESSIVENESS
                                Set aggressiveness of VAD: an integer between 0 and 3,
                                0 being the least aggressive about filtering out non-
                                speech, 3 the most aggressive. Default: 3
        --nospinner           Disable spinner
        -w SAVEWAV, --savewav SAVEWAV
                                Save .wav files of utterences to given directory
        -f FILE, --file FILE  Read from .wav file instead of microphone
        -m MODEL, --model MODEL
                                Path to the model (protocol buffer binary file, or
                                entire directory containing all standard-named files
                                for model)
        -s SCORER, --scorer SCORER
                                Path to the external scorer file.
        -d DEVICE, --device DEVICE
                                Device input index (Int) as listed by
                                pyaudio.PyAudio.get_device_info_by_index(). If not
                                provided, falls back to PyAudio.get_default_device().
        -r RATE, --rate RATE  Input device sample rate. Default: 16000. Your device
                                may require 44100.

.. _transcribe_other_options:

Other Transcription Options
---------------------------

ESPnet
^^^^^^

`espnet/espnet <https://github.com/espnet/espnet>`_ is extremely promising but is very slow for some reason. The "ASR demo" can be found in the `main README <https://github.com/espnet/espnet#asr-demo>`_.

The ESPnet commands to transcribe a WAV file are:

.. code-block:: bash

    cd egs/librispeech/asr1
    . ./path.sh
    ./../../../utils/recog_wav.sh --ngpu 1 --models librispeech.transformer.v1 example.wav

Installation can be completed with:

.. code-block:: bash

    # OS setup
    !cat /etc/os-release
    !apt-get install -qq bc tree sox

    # espnet setup
    !git clone --depth 5 https://github.com/espnet/espnet
    !pip install -q torch==1.1
    !cd espnet; pip install -q -e .

    # download pre-compiled warp-ctc and kaldi tools
    !espnet/utils/download_from_google_drive.sh \
        "https://drive.google.com/open?id=13Y4tSygc8WtqzvAVGK_vRV9GlV7TRC0w" espnet/tools tar.gz > /dev/null
    !cd espnet/tools/warp-ctc/pytorch_binding && \
        pip install -U dist/warpctc_pytorch-0.1.1-cp36-cp36m-linux_x86_64.whl

    # make dummy activate
    !mkdir -p espnet/tools/venv/bin && touch espnet/tools/venv/bin/activate
    !echo "setup done."

wav2letter
^^^^^^^^^^

Wav2letter is an "open source speech processing toolkit" written in C++ that is "built to facilitate research in end-to-end models for speech recognition." It contains pre-trained models, but the state-of-the-art models can not easily be used with the separate inference scripts. They need to be converted. The `inference tutorial <https://github.com/facebookresearch/wav2letter/wiki/Inference-Run-Examples>`_ is helpful, but it uses a smaller "example model" that does not reach state-of-the-art accuracy.

It is recommended to use wav2letter with docker due to the complex dependency tree.

The `simple_streaming_asr_example <https://github.com/facebookresearch/wav2letter/blob/master/inference/inference/examples/SimpleStreamingASRExample.cpp>`_ script can transcribe a WAV file when it is provided with the models.

The pre-trained SOTA models are `in this folder <https://github.com/facebookresearch/wav2letter/tree/master/recipes/models/sota/2019>`_ and are from the `"End-to-end ASR: from Supervised to Semi-Supervised Learning with Modern Architectures" <https://arxiv.org/abs/1911.08460>`_ paper.

This issue is currently open and disscusses the lack of clear instructions about how to use the SOTA models for inference: `Any example code using the new pretrained models <https://github.com/facebookresearch/wav2letter/issues/485>`_

It may be possible to use the ``streaming_convnets`` research models for inference if they are converted using `StreamingTDSModelConverter.cpp <https://github.com/facebookresearch/wav2letter/blob/master/tools/StreamingTDSModelConverter.cpp>`_, which has instruction `in this README <https://github.com/facebookresearch/wav2letter/tree/master/tools#streaming-tds-model-conversion-for-running-inference-pipeline>`_.
