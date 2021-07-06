.. _summarizers:

Summarization Models
====================

.. warning::
    This page discusses the actual models and algorithms used to summarize text. If you want to compare the methods available for summarizing text as used in the End-To-End process please visit :ref:`e2e_summarization_approaches`. If you are interested in the code behind the models, would like to reproduce results, or want to adapt upon the low-level summarization components, you're in the right place.

**Neural:** The extractive summarizers are provided by `HHousen/TransformerSum <https://github.com/HHousen/TransformerSum>`_ and the abstractive ones from `huggingface/transformers <https://github.com/huggingface/transformers>`_ (abstractive was originally accomplished with `HHousen/DocSum <https://github.com/HHousen/DocSum>`_). Please see those repositories for details on the exact implementation details of the models. Some of the architectures are HHousen's, some are partly HHousen's, and many are from other research projects.

**Non-Neural Algorithms:** The `sumy <https://pypi.org/project/sumy/>`_ (`Sumy GitHub <https://github.com/miso-belica/sumy>`_) package provides some non-neural summarization algorithms, mainly the methods for :meth:`~lecture2notes.end_to_end.summarization_approaches.generic_extractive_sumy` such as ``lsa``, ``luhn``, ``lex_rank``, ``text_rank``, ``edmundson``, and ``random``.

Note: The `summa <https://pypi.org/project/summa/>`_ (`Summa GitHub <https://github.com/summanlp/textrank>`_) package is used to extract keywords using the TextRank algorithm in :meth:`~lecture2notes.end_to_end.summarization_approaches.keyword_based_ext`.

.. note::
    All other models/algorithms are, to the best of my knowledge, novel and are directly implemented as part of this project. See :ref:`e2e_summarization_approaches` for details.
