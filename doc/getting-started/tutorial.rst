Tutorial
========

After you've installed lecture2notes using the instructions in :ref:`install`, you can follow this guide to perform some common actions.

.. _tutorial_general_summarize:

Summarizing a lecture video
---------------------------

.. code-block:: bash

    cd End-To-End
    python main.py --auto_id --remove_duplicates --deepspeech_model_dir ../deepspeech-models/ <path to video>

* ``--auto_id`` changes the location where files will be saved during processing to a folder in the present working directory named the first 12 characters of the input video's SHA1 hash.
* ``--remove_duplicates`` will remove duplicate slides before clustering. Once frames have been classified and the slides have been identified, there are likely to be duplicates. Setting this option removes very closely matching slides. Similar slides are detected using perceptual hashing by default (other hashing methods are available).
* ``--deepspeech_model_dir`` is the path to the folder containing the DeepSpeech model files (the ``.pbmm`` acoustic model and the scorer).
* ``<path to video>`` is the path to your video file that you want to process.

Summarizing a lecture video on YouTube
--------------------------------------

1. First, download the video from YouTube: ``youtube-dl <video url>`` (where ``<video url>`` looks like ``https://www.youtube.com/watch?v=dQw4w9WgXcQ``)
2. Then, process the file using a command similar to the one in :ref:`tutorial_general_summarize`:

    .. code-block:: bash

        cd End-To-End
        python main.py --auto_id --remove_duplicates --deepspeech_model_dir ../deepspeech-models/ --transcription_method youtube --video_id <video id from youtube> <path to video>

The only differences from :ref:`tutorial_general_summarize` are the addition of ``--transcription_method youtube`` and ``--video_id <video id from youtube>``. 

* ``--transcription_method youtube`` makes the script attempt to download the transcript of the video directly from YouTube. This only works when the video has manually attached captions on YouTube. Videos with automatically generated captions from YouTube will still be processed locally using DeepSpeech by default. This is why the ``--deepspeech_model_dir`` argument is still specified.
* ``--video_id <video id from youtube>`` tells the script which video from youtube to download the captions from. ``<video id from youtube>`` should be set to the id of the video. The id is the part after the ``?v=`` in the url up to (and not including) the ``&`` or if no ``&`` then until the end of the url (if the video link is ``https://www.youtube.com/watch?v=dQw4w9WgXcQ`` then the id is ``dQw4w9WgXcQ``).
