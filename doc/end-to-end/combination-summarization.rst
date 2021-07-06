.. _e2e_summarization_approaches:

Combination and Summarization
=============================

Once the system has a plain text representation of the visual and auditory information portrayed in the lecture, it can begin to summarize and create the final notes. We break this into a four-stage process: combination, modification, extractive summarization, and abstractive summarization. Some of these steps can be turned off completely. For example, it is possible to combine the two transcripts and then summarize the result using an abstractive model, thus skipping the modifications and extractive summarization steps. The word "transcripts" in this section refers to the audio transcript and text content extracted from the slides.

.. _structured_joined_summarization:

Structured Joined Summarization
-------------------------------

The structured joined summarization method is separate from the four-stage process. This method summarizes slides using the SSA (see :ref:`slide_structure_analysis`) and audio transcript to create an individual summary for each slide. The words spoken from the beginning of one slide to the start of the next to the nearest sentence are considered the transcript that belongs to that slide. Either DeepSpeech, Vosk, or manual transcription must be used to summarize using ``structured_joined`` because they are the only models that output the start times of each word spoken. The transcript that belongs to each slide is independently summarized using extractive (see :ref:`e2e_extractive_summarization`) or abstractive summarization (see :ref:`e2e_abstractive_summarization`). The content from each slide is formatted following the SSA and presented to the user. Only paragraphs on the slide longer than three characters are included. The result contains the following for each unique slide identified: a summary of the words spoken while that slide was displayed, the formatted content on the slide, an image of the slide, and extracted figures from the slide.

You can learn more in the docstring for the :meth:`lecture2notes.end_to_end.summarization_approaches.structured_joined_sum` function.

.. _e2e_combination:

Combination
-----------

There are five methods of combining the transcripts:

1. ``only_asr``: only uses the audio transcript (deletes the slide transcript)
2. `only_slides`: only uses the slides transcript (deletes the audio transcript)
3. ``concat``: appends audio transcript to slide transcript
4. ``full_sents``: audio transcript is appended to only the complete sentences of the slide transcript
5. ``keyword_based``: selects a certain percentage of sentences from the audio transcript based on keywords found in the slides transcript

Full Sentences Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^

Complete sentences are detected by tokenizing the input text using a Spacy model and then selecting sentences that end with punctuation and contain at least one subject, predicate, and object (two nouns and one verb). If the number of tokens in the text containing only complete sentences is greater than a percentage of the number of tokens in the original text then the algorithm returns the complete sentences, otherwise, it will return the original text. This check ensures that a large quantity of content is not lost. By default, the cutoff percentage is 70\%. You can view the code here: :meth:`lecture2notes.end_to_end.summarization_approaches.full_sents`.

.. _combination_keyword_based:

Keyword Based Algorithm
^^^^^^^^^^^^^^^^^^^^^^^

Since the text extracted from slides may contain many incomplete ideas that the presenter clarifies through speech, it may be necessary to completely disregard the slide transcript. However, in the case where the slide transcript contains a loose connection of ideas we view it as an incomplete summary of the lecture. Thus, the information in the slide transcript can be used to select the most important sentences from the audio transcript, thus preventing the loss of significant amounts of information.

First, keywords are extracted from the slide transcript using TextRank. Next, this list of keywords is used as the vocabulary in the creation of a TF-IDF (term frequency-inverse document frequency) vectorizer. The TF-IDF vectorizer is equivalent to converting the voice transcript to a matrix of token counts followed by performing the TF-IDF transformation. After the TF-IDF vectorizer is created, the sentences are extracted from the transcript text using a Spacy model. Finally, the document term matrix is created by fitting the TF-IDF vectorizer on the sentences from the transcript text and then transforming the transcript text sentences using the fitted vectorizer. Next, the algorithm calculates the singular value decomposition (SVD), which is known as latent semantic analysis (LSA) in the context of TF-IDF matrices, of the document term matrix. To compute the ranks of each sentence we pass :math:`\Sigma` and :math:`V` to the rank computation algorithm, which for each row vector in the transposed :math:`V` matrix finds the sum of :math:`s^2*v^2` where :math:`s` is the row of :math:`\Sigma` and :math:`v` is the column of :math:`V`. Finally, the algorithm selects the sentences in the top 70\% sorted by rank. The sentences are sorted by their original position in the document to ensure the order of the output sentences follows their order in the input document.


.. _e2e_modifications:

Modifications
-------------

Modifications change the output of the combination algorithm before it is sent to the summarization stages. The only modification is a function that will extract the complete sentences from the combined audio and slide transcript. By default, this modification is turned off. Importantly, the modification framework is extendable so that future modifications can be implemented easily.


.. _e2e_extractive_summarization:

Extractive Summarization
------------------------

There are two high-level extractive summarization methods: ``cluster``, an advanced novel algorithm, and ``generic``, which is a collection of several standard extractive summarization algorithms such as TextRank, LSA, and Edmundson.

The Cluster Method
^^^^^^^^^^^^^^^^^^

The ``cluster`` algorithm extracts features from the text, clusters based on those features, and summarizes each cluster.

Feature extraction can be done in four ways:

1. ``neural_hf``: Uses a large transformer model (RoBERTa by default) to extract features.
2. ``neural_sbert``: Uses special BERT and RoBERTa models fine-tuned to extract sentence embeddings. This is the default option.
3. ``spacy``: Uses the ``en_core_web_lg`` (large model is preferred over a smaller model since large models have "read" word vectors) Spacy model to loop through sentences and store their "vector" values, which is an average of the token vectors.
4. ``bow``: The name "bow" stands for "bag of words." This method is fast since it is based on word frequencies throughout the input text. The implementation is similar to the combination keyword based algorithm (see :ref:`combination_keyword_based`) but instead of using keywords from another document, the keywords are calculated using the TF-IDF vectorizer. The TF-IDF-weighted document-term matrix contains the features that are clustered.

The feature vectors are clustered using the KMeans algorithm. The user must specify the number of clusters they want, which corresponds to the number of topics discussed in the lecture. Mini batch KMeans is supported if a reduction in computation time is desired and obtaining slightly worse results is acceptable.

Summarization can be done in two ways:

1. ``extractive``: Computes the SVD of the document-term matrix and calculates ranks using the sentence rank computation algorithm. This option requires that features were extracted using ``bow`` because this method needs the document-term matrix produced during ``bow`` feature extraction in order to compute sentence rankings.
2. ``abstractive``: Summarizes text using a seq2seq transformer model trained on abstractive summarization. The default model is a distilled version of BART.

The clusters are summarized sequentially. However, when using the ``extractive`` summarization method, the ranks of all sentences are only calculated once before clustering. Before summarization, the sentences and corresponding ranks are grouped by cluster centroid. The TF-IDF and ranks are calculated at the document level, not the cluster level.

Automatic Cluster Title Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There is an additional optional stage of the ``cluster`` extractive summarization method that will create titles for each cluster. This is accomplished by summarizing each cluster twice: once for the actual summary and again to create the title. Since titles are much shorter than the content, a seq2seq transformer trained on the XSum dataset is used (BART is the default). XSum contains documents and one-sentence news summaries answering the question "What is the article about?". To encourage short titles, when decoding the model output of the cluster we set the minimum length to one token and the maximum to ten tokens. This produces subpar titles but, the structured joined summarization method (see :ref:`structured_joined_summarization`) solves this problem.

The Generic Method
^^^^^^^^^^^^^^^^^^

There are six generic extractive summarization methods: ``lsa``, ``luhn``, ``lex_rank``, ``text_rank``, ``edmundson``, and ``random``. Random selects sentences from the input document at random.

.. _e2e_abstractive_summarization:

Abstractive Summarization
-------------------------

The abstractive summarization stage is applied to the result of the extractive summarization stage. If extractive summarization was disabled then abstractive summarization simply transforms the result of the modifications stage.

This stage of the summarization steps passes the text of the previous step through a seq2seq transformer model and returns the result. Multiple seq2seq models can be used to summarize including `TransformerSum <https://github.com/HHousen/TransformerSum/>`_, `BART <https://huggingface.co/transformers/model_doc/bart.html>`_, `PEGASUS <https://huggingface.co/transformers/model_doc/pegasus.html>`_, `T5 <https://huggingface.co/transformers/model_doc/t5.html>`_, `PreSumm <https://arxiv.org/abs/1908.08345>`_ (`how TransformerSum improves PreSumm <https://transformersum.readthedocs.io/en/latest/general/differences.html>`_).
