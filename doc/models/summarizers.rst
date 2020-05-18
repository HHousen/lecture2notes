Summarizers
===========

The extractive summarizers are provided by `HHousen/TransformerExtSum <https://github.com/HHousen/TransformerExtSum>`_ and the abstractive ones from `HHousen/DocSum <https://github.com/HHousen/DocSum>`_. Please see those repositories for details on the exact implementation details of the models. Some of the architectures are HHousen's, some are partly HHousen's, and many are from other research projects.

The `sumy <https://pypi.org/project/sumy/>`_ (`GitHub <https://github.com/miso-belica/sumy>`_) package provides the methods for ``generic_extractive_sumy()`` in ``End-To-End/summarization_approaches.py`` such as ``lsa``, ``luhn``, ``lex_rank``, ``text_rank``, ``edmundson``, and ``random``.