# Lecture2Notes

> Convert lecture videos to notes using AI & machine learning.

[![GitHub license](https://img.shields.io/github/license/HHousen/lecture2notes.svg)](https://github.com/HHousen/lecture2notes/blob/master/LICENSE) [![Github commits](https://img.shields.io/github/last-commit/HHousen/lecture2notes.svg)](https://github.com/HHousen/lecture2notes/commits/master) [![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) [![Documentation Status](https://readthedocs.org/projects/lecture2notes/badge/?version=latest)](https://lecture2notes.readthedocs.io/en/latest/?badge=latest) [![GitHub issues](https://img.shields.io/github/issues/HHousen/lecture2notes.svg)](https://GitHub.com/HHousen/lecture2notes/issues/) [![GitHub pull-requests](https://img.shields.io/github/issues-pr/HHousen/lecture2notes.svg)](https://GitHub.com/HHousen/lecture2notes/pull/) [![DeepSource](https://static.deepsource.io/deepsource-badge-light-mini.svg)](https://deepsource.io/gh/HHousen/lecture2notes/?ref=repository-badge)

**[Research Paper]()** / **[Documentation](https://lecture2notes.readthedocs.io/en/latest)** / **[Project Page](https://haydenhousen.com/projects/lecture2notes/)**

**Lecture2Notes** is a project that **summarizes lectures videos**. At a high level, it parses both the **visual** and **auditory** components of the video, **extracts text** from each, **combines** them, and then **summarizes** the combined text using **automatic** summarization algorithms. These pages document the code for the entirety of "[Lecture2Notes: Summarizing Lecture Videos by Classifying Slides and Analyzing Text using Machine Learning]()."

**Check out [the documentation](https://lecture2notes.readthedocs.io/en/latest) for usage details.**

**To get started summarizing text**, visit [the tutorial](https://lecture2notes.readthedocs.io/en/latest/getting-started/tutorial.html).

## Abstract

Note-taking is a universal activity among students because of its benefits to the learning process. This research focuses on end-to-end generation of formatted summaries of lecture videos. Our automated multimodal approach will decrease the time required to create notes, increase quiz scores and content knowledge, and enable faster learning through enhanced previewing. The project is broken into three main components: the slide classifier, summarization models, and end-to-end-process. The system beings by extracting important keyframes using the slide classifier, a deep CNN. Then, unique slides are determined using a combination of clustering and keypoint matching. The structure of these unique slides is analyzed and converted to a formatted transcript that includes figures present on the slides. The audio is transcribed using one of several methods. We approach the process of combining and summarizing these transcripts in several ways including as keyword-based sentence extraction and temporal audio-slide-transcript association problems. For the summarization stage, we created TransformerSum, a summarization training and inference library that advances the state-of-the-art in long and resource-limited summarization, but other state-of-the-art models, such as BART or PEGASUS, can be used as well. Extractive and abstractive approaches are used in conjunction to summarize the long-form content extracted from the lectures. While the end-to-end process and each individual component yield promising results, key areas of weakness include the speech-to-text algorithm failing to identify certain words and some summarization methods producing sub-par summaries. These areas provide opportunities for further research.

## Details

The project is broken into four main components: the [slide classifier](https://lecture2notes.readthedocs.io/en/latest/models/slide-classifier.html#sc-overview) (including the [dataset](https://lecture2notes.readthedocs.io/en/latest/dataset/general.html#dataset-general-information)), the [summarization models](https://lecture2notes.readthedocs.io/en/latest/models/summarizers.html#summarizers) (neural, non-neural, extractive, and abstractive), the [end-to-end-process](https://lecture2notes.readthedocs.io/en/latest/end-to-end/general.html#e2e-general-info) (one command to convert to notes), and finally the [website](https://lecture2notes.com>) that enables users to process their own videos.

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

9. Convert intermediate outputs to a final notes file (HTML, TXT, markdown, etc.)

The summarization steps can be toggled off and on (see [Combination and Summarization](https://lecture2notes.readthedocs.io/en/latest/end-to-end/combination-summarization.html#e2e-summarization-approaches)).

## Meta

[![ForTheBadge built-with-love](https://ForTheBadge.com/images/badges/built-with-love.svg)](https://GitHub.com/HHousen/)

Hayden Housen – [haydenhousen.com](https://haydenhousen.com)

Distributed under the GNU Affero General Public License v3.0 (AGPL). See the [LICENSE](LICENSE) for more information.

<https://github.com/HHousen>

## Contributing

1. Fork it (<https://github.com/HHousen/lecture2notes/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request
