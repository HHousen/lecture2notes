About
=====

Overview
--------

**Lecture2Notes** is a project that **summarizes lectures videos**. At a high level, it parses both the **visual** and **auditory** components of the video, **extracts text** from each, **combines** them, and then **summarizes** the combined text using **automatic** summarization algorithms. These pages document the code for the entirety of "`Lecture2Notes: Summarizing Lecture Videos by Classifying Slides and Analyzing Text using Machine Learning <https://haydenhousen.com/media/lecture2notes-paper-v1.pdf>`__."

**To get started**, visit :ref:`the tutorial <tutorial_general_summarize>`.

**Visit `lecture2notes.com <https://lecture2notes.com/>`__ to convert your own lectures to notes!**

The project is broken into four main components: the :ref:`slide classifier <sc_overview>` (including the :ref:`dataset <dataset_general_information>`), the :ref:`summarization models <summarizers>` (neural, non-neural, extractive, and abstractive), the :ref:`end-to-end-process <e2e_general_info>` (one command to convert to notes), and finally the `website <https://lecture2notes.com>`_ that enables users to process their own videos.

Process:
    1. Extract frames from video file
    2. Classify extracted frames to find frames containing slides
    3. Perspective crop images containing the presenter and slide to contain only the slide by matching temporal features
    4. Cluster slides to group transitions and remove duplicates
    5. Run a Slide Structure Analysis (SSA) using OCR on the slide frames to obtain a formatted transcript of the text on the slides
    6. Detect and extract figures from the set of unique slide frames
    7. Transcribe the lecture using a speech-to-text algorithm
    8. Summarize the visual and auditory transcripts

       1. Combine
       2. Run some modifications (such as only using complete sentences)
       3. Extractive summarization
       4. Abstractive summarization

    9.  Convert intermediate outputs to a final notes file (HTML, TXT, markdown, etc.)

The summarization steps can be toggled off and on (see :ref:`e2e_summarization_approaches`).

.. note::
    Notice an issue with this documentation? A typo? Missing or incorrect information? Please open an issue on `GitHub <https://github.com/HHousen/lecture2notes>`_ or click "Edit on GitHub" in the top right corner of the page with issues. Documentation is almost as important as code (what's the point of having code that can't be understood). Please report problems you find (even if its one letter). Thanks.

**Abstract:** Note-taking is a universal activity among students because of its benefits to the learning process. This research focuses on end-to-end generation of formatted summaries of lecture videos. Our automated multimodal approach will decrease the time required to create notes, increase quiz scores and content knowledge, and enable faster learning through enhanced previewing. The project is broken into three main components: the slide classifier, summarization models, and end-to-end-process. The system beings by extracting important keyframes using the slide classifier, a deep CNN. Then, unique slides are determined using a combination of clustering and keypoint matching. The structure of these unique slides is analyzed and converted to a formatted transcript that includes figures present on the slides. The audio is transcribed using one of several methods. We approach the process of combining and summarizing these transcripts in several ways including as keyword-based sentence extraction and temporal audio-slide-transcript association problems. For the summarization stage, we created TransformerSum, a summarization training and inference library that advances the state-of-the-art in long and resource-limited summarization, but other state-of-the-art models, such as BART or PEGASUS, can be used as well. Extractive and abstractive approaches are used in conjunction to summarize the long-form content extracted from the lectures. While the end-to-end process and each individual component yield promising results, key areas of weakness include the speech-to-text algorithm failing to identify certain words and some summarization methods producing sub-par summaries. These areas provide opportunities for further research.


Components
----------

1. Slide Classifier
    * **Overview:** The slide classifier is a computer vision machine learning model that classifies images into 9 categories as listed on :ref:`its documentation page <sc_overview>`.
    * **Key Info:** Most importantly are the ``slide`` and ``presenter_slide`` categories which refer to slides from a screen capture and slides as recorded by a video camera, respectively. When a video camera is pointed at the slides, the frame will usually include the presenter, which is the reasoning behind the name choices. Frames from the ``slide`` class are processed differently than those from the ``presenter_slide`` class. Namely, those from ``presenter_slide`` are automatically perspective cropped while those from ``slide`` are not.
    * **Dataset:** The dataset was collected using the scraper scripts in ``dataset/scraper-scripts``. To learn about how to collect the dataset visit :ref:`dataset_general_walkthrough`. You can view information about each script in :ref:`ss_home`.
2. Summarization Models
    * **Locations:** The neural summarization models are located in ``models`` while the non-neural algorithms are implemented in :ref:`e2e_summarization_approaches` (``end_to_end/summarization_approaches``).
    * **Neural Extractive Models:** https://github.com/HHousen/TransformerSum
    * **Neural Abstractive Models:** https://github.com/huggingface/transformers & https://github.com/HHousen/DocSum
    * **More Info:** See :ref:`summarizers`.
3. End-To-End Process
    * **Overview:** Brings everything together to summarize lecture videos. It requires only one command to summarize a lecture video. That command can contain 20 arguments or only 1: the path to the file. See :ref:`the tutorial <tutorial_general_summarize>`.
    * **API Documentation:** :ref:`e2e_api`, use if you want to modify the scripts, if you want to write new components (`pull requests welcome <https://github.com/HHousen/lecture2notes/compare>`_), or if you want to use certain components programmatically (:ref:`guide to programmatically summarize a lecture <tutorial_programmatically>`).
    * **General Info:** :ref:`e2e_general_info`, use if you want to fine-tune the parameters used for conversion.
    * **Summarization Approaches:** :ref:`e2e_summarization_approaches`, specific information about how the lecture is summarized
4. Website
    * https://lecture2notes.com

The directory structure of the project should be relatively easy to follow. There is essentially a subfolder in the ``lecture2notes`` folder for each major component discussed above (documentation is in ``docs/`` at the root level of the repository).

.. note::
    The slide classifier dataset is located in ``dataset`` and the model is located in ``models/slide_classifier``. This separation was made to disconnect the data collection code from the model training code, since they are two distinct stages of the process that require little interaction (the only interaction is the copying of the final dataset).

* ``dataset``: Data collection code for the slide classifier.
* ``end_to_end``: Contains all the code (except :py:mod:`lecture2notes.models.slide_classifier.inference` and some summarization models) required to summarize a lecture video. This includes frame extraction, OCR, clustering, perspective cropping, spell checking, speech to text, and more.
* ``models``: Contains the slide classifier model training code and the legacy neural summarization model repository (https://github.com/HHousen/DocSum/) as a git module.

FAQ
---

Want to add to the FAQ? Open an issue on GitHub or click "Edit on GitHub" above. All contributions are greatly appreciated. If you're asking it, someone else probably is too.

Where are the summarization models?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TL;DR: https://github.com/HHousen/TransformerSum

The neural-based summarization models that were created as a major component of this research are not part of this repository. While initially developed as part of this repository, they were broken off due to the complexity of the code and the applicability to future projects. You can view and run the training code and use 10+ pre-trained models at https://github.com/HHousen/TransformerSum. Essentially, the models are more accessible to other researchers for projects unrelated to lectures if they reside in their own repository.

See :ref:`summarizers` for more information.

Significant People
------------------

The project was created by `Hayden Housen <https://haydenhousen.com/>`_ during his sophomore, junior, and seniors years of high school as part of the Science Research program. It is actively maintained and updated by him and the community.
