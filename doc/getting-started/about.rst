About
=====

Overview
--------

Lecture2Notes is a project that summarizes lectures videos. At a high level, it parses both the visual and auditory components of the video, extracts text from each, combines them, and then summarizes the combined text using automatic summarization algorithms. These pages document the code for the entirety of [Research Paper Name].

The project is broken into four main components: the slide classifier (including the dataset), the summarization models (neural, non-neural, extractive, and abstractive), the end-to-end-process (one command to convert to notes), and finally the website that enables users to process their own videos.

Process:
    1. Extract frames from video file
    2. Classify extracted frames to find frames with slides
    3. Perspective crop images of ``presenter_slide`` class to contain only the slide
    4. Cluster slides group transitions and remove duplicates
    5. Run OCR on the slide frames to obtain a transcript of the text on the slides
    6. Transcribe the lecture using a speech-to-text algorithm
    7. Summarize the visual and auditory transcripts
        1. Combine
        2. Run some modifications (such as only using complete sentences)
        3. Extractive summarization
        4. Abstractive summarization

The summarization steps can be toggled off and on (see :ref:`e2e_summarization_approaches`).

.. note::
    Notice an issue with this documentation? A typo? Missing or incorrect information? Please open an issue on `GitHub <https://github.com/HHousen/lecture2notes>`_ or click "Edit on GitHub" in the top right corner of the page with issues. Documentation is almost as important as code (what's the point of having code that can't be understood). Please report problems you find (even if its one letter). Thanks.

**Abstract:** Note-taking is a universal activity among students. Students take notes during lectures to force the active interpretation of the information they are learning. This research focuses on applying extractive and abstractive summarization techniques to transcripts of the visual and auditory content of lectures to create detailed notes. This automated multimodal approach will decrease the time required to create notes, increase quiz scores and content knowledge, and enable faster learning through enhanced previewing. The project is broken into four main components: the slide classifier, the summarization models, the end-to-end-process, and finally the website that enables users to process their videos. The slide classifier is an EfficientNet, although other architectures were tested, trained on tens of thousands of frames from lecture videos. Google's Tesseract project is used to perform OCR on the identified slides and Mozilla's implementation of Baidu's DeepSpeech is used to transcribe the audio.  The process of combining these transcripts is novel and can be accomplished in several ways. For the summarization stage, state-of-the-art models are used, including BART, PreSumm, and novel models specifically for this project, which are collectively called "TransformerExtSum." Extractive and abstractive approaches are combined to summarize the long-form content extracted from the lectures. While the end-to-end process yields promising results, key areas of weakness include the speech-to-text algorithm failing to identify certain words and the summarization models producing sub-par summarizes. These areas provide opportunities for further research.

Components
----------

1. Slide Classifier
    * **Overview:** The slide classifier is a computer vision machine learning model that classifies images into 9 categories as listed on :ref:`its documentation page <sc_overview>`. 
    * **Key Info:** Most importantly are the ``slide`` and ``presenter_slide`` categories which refer to slides from a screen capture and slides as recorded by a video camera, respectively. When a video camera is pointed at the slides, the frame will usually include the presenter, which is the reasoning behind the name choices. Frames from the ``slide`` class are processed differently than those from the ``presenter_slide`` class. Namely, those from ``presenter_slide`` are automatically perspective cropped while those from ``slide`` are not.
    * **Dataset:** The dataset was collected using the scraper scripts in ``Dataset/scraper-scripts``. To learn about how to collect the dataset visit :ref:`dataset_general_walkthrough`. You can view information about each script in :ref:`ss_home`.
2. Summarization Models
    * **Locations:** The neural summarization models are located in ``Models`` while the non-neural algorithms are implemented in :ref:`e2e_summarization_approaches` (``End-To-End/summarization_approaches``).
    * **Neural Extractive Models:** https://github.com/HHousen/TransformerExtSum
    * **Neural Abstractive Models:** https://github.com/HHousen/DocSum
    * **More Info:** See :ref:`summarizers`.
3. End-To-End Process
    * **Overview:** Brings everything together to summarize lecture videos. It requires only one command to summarize a lecture video. That command can contain 20 arguments or only 1: the path to the file. See :ref:`the tutorial <tutorial_general_summarize>`.
    * **API Documentation:** :ref:`e2e_api`, use if you want to modify the scripts or if you want to write new components (`pull requests welcome <https://github.com/HHousen/lecture2notes/compare>`_)
    * **General Info:** :ref:`e2e_general_info`, use if you want to finetune the parameters used for conversion.
    * **Summarization Approaches:** :ref:`e2e_summarization_approaches`, specific information about how the lecture is summarized
4. Website
    * Coming soon...

The directory structure of the project should be relatively easy to follow. There is essentially a subfolder in the project root for each major component discussed above (and the documentation). 

.. note::
    The slide classifier dataset is located in ``Dataset`` and the model is located in ``Models/slide-classifier``. This separation was made to disconnect the data collection code from the model training code, since they are two distinct stages of the process that require little interaction (the only interaction is the copying of the final dataset).

* ``Dataset``: Data collection code for the slide classifier.
* ``End-To-End``: Contains all the code (except ``Models/slide-classifier/inference.py`` and some summarization models) required to summarize a lecture video. This includes frame extraction, OCR, clustering, perspective cropping, spell checking, speech to text, and more.
* ``Models``: Contains the slide classifier model training code and the neural summarization model repositories as git modules.

FRQ
---

Want to add to the FRQ? Open an issue on GitHub or click "Edit on GitHub" above. All contributions are greatly appreciated. If you're asking it, someone else probably is too.

Where are the summarization models?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The neural-based summarization models, while a major component of this research, are not part of this repository. While initially developed as part of this repository, they were broken off due to the complexity of the code and the applicability to future projects. Essentially, the models are more accessible to other researchers in their current state.

See :ref:`summarizers` for more information.

Significant People
------------------

The project was created by `Hayden Housen <https://haydenhousen.com/>`_ during his sophomore year of highschool as part of the Science Research program. It is actively maintained and updated by him and the community.