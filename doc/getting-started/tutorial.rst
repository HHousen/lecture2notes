Tutorial
========

After you've installed lecture2notes using the instructions in :ref:`install`, you can follow this guide to perform some common actions.

..note:: Read `the paper <https://haydenhousen.com/media/lecture2notes-paper-v1.pdf>`__ for more in-depth explanations regarding the background, methodology, and results of this project.

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

.. _tutorial_programmatically:

Summarizing a lecture video programmatically
--------------------------------------------

Summarizing a lecture video programmatically is fairly easy due to the :class:`~lecture2notes.end_to_end.summarizer_class.LectureSummarizer` class.

First, create a dictionary or ``argparse.Namespace`` of the settings you want to use for your :class:`~lecture2notes.end_to_end.summarizer_class.LectureSummarizer` object. The default parameters are stored in a json file at ``end_to_end/default_params.json``. This file directly mirrors the defaults set for each argparse argument in ``end_to_end/main.py``, the script used for CLI usage.

However, the default configuration leaves ``video_path`` set to null so we need to override this. Instead of modifying or creating a new configuration object for each video, you can pass overrides to the :class:`~lecture2notes.end_to_end.summarizer_class.LectureSummarizer` object upon creation (see below).

Next, create your :class:`~lecture2notes.end_to_end.summarizer_class.LectureSummarizer` object, call the :meth:`~lecture2notes.end_to_end.summarizer_class.LectureSummarizer.run_all` function, and get the summary like so:

.. code-block:: python

    from lecture2notes.end_to_end.summarizer_class import LectureSummarizer
    default_config_path = "lecture2notes/end_to_end/default_params.json"
    video_path = "path/to/my/amazing/lecture/video.mp4"
    summarizer = LectureSummarizer(default_config_path, video_path=video_path)

    summarizer.run_all()

    structured_summary = summarizer.final_data["structured_summary"]
    lecture_summary = summarizer.final_data["lecture_summary"]
    transcript = summarizer.final_data["transcript"]

Alternatively, you can iterate over the ``all_step_functions`` attribute of your :class:`~lecture2notes.end_to_end.summarizer_class.LectureSummarizer` object to run your own code between each step of the process. For example, you can store the current step in a database or to the file system so that if you restart your program the :class:`~lecture2notes.end_to_end.summarizer_class.LectureSummarizer` can automatically resume:

.. code-block:: python

    last_step_run = int(open("last_step_run.txt", "r").read())

    with open("last_step_run.txt", "w") as file:
        for idx, step_func in enumerate(summarizer.all_step_functions):
            if idx + 1 < last_step_run:
                # Skip steps that have already been ran
                continue

            last_step_run = idx + 1
            file.write(last_step_run)

            step_func()
