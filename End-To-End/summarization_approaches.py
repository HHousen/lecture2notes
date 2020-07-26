import logging
import sys
import json
import os
import spacy
import math
from time import time
from tqdm import tqdm
from collections import namedtuple, defaultdict, OrderedDict
from operator import attrgetter, itemgetter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Sklearn imports for cluster()
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans

# Sumy Imports for generic_extractive_sumy()
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.edmundson import EdmundsonSummarizer
from sumy.summarizers.random import RandomSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

# Import transformers and sentence_transformers for neural based feature extraction from text
from transformers import pipeline
from sentence_transformers import SentenceTransformer

sys.path.insert(1, os.path.join(os.getcwd(), '../Models/summarizer'))

logger = logging.getLogger(__name__)

def get_complete_sentences(text, return_string=False):
    nlp = spacy.load("en_core_web_sm")
    complete_sentences = list()

    text = text.replace('\n', ' ').replace('\r', '')
    
    NLP_DOC = nlp(text)
    NUM_TOKENS = len(NLP_DOC)
    NLP_SENTENCES = list(NLP_DOC.sents)
    
    # Detect Complete Sentences:
    # A complete sentence contains at least one subject, one predicate, one object, and closes
    # with punctuation. Subject and object are almost always nouns, and the predicate is always
    # a verb. Thus to check if a sentence is a complete sentence, check if it contains two nouns,
    # one verb and closes with punctuation.
    # https://stackoverflow.com/questions/50454857/determine-if-a-text-extract-from-spacy-is-a-complete-sentence
    for sent in NLP_SENTENCES:
        if sent[0].is_title and sent[-1].is_punct:
            has_noun = 2
            has_verb = 1
            for token in sent:
                if token.pos_ in ["NOUN", "PROPN", "PRON"]:
                    has_noun -= 1
                elif token.pos_ == "VERB":
                    has_verb -= 1
            if has_noun < 1 and has_verb < 1:
                complete_sentences.append(sent)
    if return_string:
        return NUM_TOKENS, " ".join(complete_sentences)
    else:
        return NUM_TOKENS, complete_sentences

def full_sents(ocr_text, transcript_text, remove_newlines=True, cut_off=0.70):
    OCR_NUM_TOKENS, complete_sentences = get_complete_sentences(ocr_test)

    OCR_NLP_SENTENCES_LENGTHS = [len(sentence) for sentence in complete_sentences]
    OCR_NLP_SENTENCES_TOT_NUM_TOKENS = sum(OCR_NLP_SENTENCES_LENGTHS)

    # Ratio of tokens in complete sentences to total number of token in document
    # cst_to_dt = complete sentence tokens to document tokens
    cst_to_dt_ratio = OCR_NLP_SENTENCES_TOT_NUM_TOKENS / OCR_NUM_TOKENS

    logger.debug("Tokens in complete sentences: " + str(OCR_NLP_SENTENCES_TOT_NUM_TOKENS) + " | Document tokens: " + str(OCR_NUM_TOKENS) + " | Ratio: " + str(cst_to_dt_ratio))

    if cst_to_dt_ratio > cut_off: # `cut_off`% of doc is complete sentences
        complete_sentences_string = " ".join(complete_sentences)
        return complete_sentences_string + transcript_text # use complete sentences and transcript
    else: # ratio does not meet `cut_off`
        return transcript_text # only use transcript

def compute_ranks(sigma, v_matrix):
    MIN_DIMENSIONS = 3
    REDUCTION_RATIO = 1/1

    assert len(sigma) == v_matrix.shape[0], "Matrices should be multiplicable"

    dimensions = max(MIN_DIMENSIONS, int(len(sigma)*REDUCTION_RATIO))
    powered_sigma = tuple(s**2 if i < dimensions else 0.0 for i, s in enumerate(sigma))

    ranks = []
    # iterate over columns of matrix (rows of transposed matrix)
    for column_vector in v_matrix.T:
        rank = sum(s*v**2 for s, v in zip(powered_sigma, column_vector))
        ranks.append(math.sqrt(rank))

    return ranks

def get_best_sentences(sentences, count, rating, *args, **kwargs):
    SentenceInfo = namedtuple("SentenceInfo", ("sentence", "order", "rating",))
    rate = rating
    if isinstance(rating, list):
        assert not args and not kwargs
        rate = lambda o: rating[o]

    infos = (SentenceInfo(s, o, rate(o, *args, **kwargs)) for o, s in enumerate(sentences))

    # sort sentences by rating in descending order
    infos = sorted(infos, key=attrgetter("rating"), reverse=True)
    # get `count` first best rated sentences
    infos = infos[:count]
    # sort sentences by their order in document
    infos = sorted(infos, key=attrgetter("order"))

    return tuple(i.sentence for i in infos)

def get_sentences(text, model="en_core_web_sm"):
    logger.debug("Tokenizing text...")
    nlp = spacy.load(model)
    NLP_DOC = nlp(text)
    logger.debug("Text tokenized successfully")
    NLP_SENTENCES = [str(sentence) for sentence in list(NLP_DOC.sents)]
    NLP_SENTENCES_SPAN = list(NLP_DOC.sents)
    NLP_SENTENCES_LEN = len(NLP_SENTENCES)
    NLP_SENTENCES_LEN_RANGE = range(NLP_SENTENCES_LEN)

    return NLP_DOC, NLP_SENTENCES, NLP_SENTENCES_SPAN, NLP_SENTENCES_LEN, NLP_SENTENCES_LEN_RANGE

def keyword_based_ext(ocr_text, transcript_text, coverage_percentage=0.70):
    from summa import keywords
    ocr_text = ocr_text.replace('\n', ' ').replace('\r', '')

    ocr_keywords = keywords.keywords(ocr_text)
    ocr_keywords = ocr_keywords.splitlines()
    logger.debug("Number of keywords: " + str(len(ocr_keywords)))

    vocab = dict(zip(ocr_keywords, range(0, len(ocr_keywords))))

    vectorizer = TfidfVectorizer(vocabulary=vocab, stop_words="english")

    _, NLP_SENTENCES, _, NLP_SENTENCES_LEN, _ = get_sentences(transcript_text)

    NUM_SENTENCES_IN_SUMMARY = int(NLP_SENTENCES_LEN * coverage_percentage)
    logger.debug(str(NLP_SENTENCES_LEN) + " (Number of Sentences in Doc) * " + str(coverage_percentage) + " (Coverage Percentage) = " + str(NUM_SENTENCES_IN_SUMMARY) + " (Number of Sentences in Summary)")

    doc_term_matrix = vectorizer.fit_transform(NLP_SENTENCES)
    logger.debug("Vectorizer successfully fit")
    # vectorizer.get_feature_names() is ocr_keywords
    doc_term_matrix = doc_term_matrix.toarray()

    doc_term_matrix = doc_term_matrix.transpose(1, 0) # flip axes so the sentences (documents) are the columns and the terms are the rows

    u, sigma, v = np.linalg.svd(doc_term_matrix, full_matrices=False)
    logger.debug("SVD successfully calculated")

    ranks = iter(compute_ranks(sigma, v))
    logger.debug("Ranks calculated")

    sentences = get_best_sentences(NLP_SENTENCES, NUM_SENTENCES_IN_SUMMARY, lambda s: next(ranks))
    logger.debug("Top " + str(NUM_SENTENCES_IN_SUMMARY) + " sentences found")

    return " ".join(sentences) # return as string with space between each sentence

def extract_features_bow(data, return_lsa_svd=False, use_hashing=False, use_idf=True, n_features=10000,
                         lsa_num_components=False):
    """Extract features using a bag of words statistical word-frequency approach. 
    
    Arguments:
        data (list): List of sentences to extract features from
        return_lsa_svd (bool, optional): Return the features and ``lsa_svd``. See "Returns"
            section below. Defaults to False.
        use_hashing (bool, optional): Use a HashingVectorizer instead of a CountVectorizer. Defaults to False.
            A HashingVectorizer should only be used with large datasets. Large to the
            degree that you'll probably never pass enough data through this function
            to warrent the usage of a HashingVectorizer. HashingVectorizers use very
            little memory and are thus scalable to large datasets because there is no 
            need to store a vocabulary dictionary in memory.
            More information can be found in the `HashingVectorizer scikit-learn documentation <https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html>`_.
        use_idf (bool, optional): Option to use inverse document-frequency. Defaults to True. In the case of ``use_hasing``
            a TfidfTransformer will be appended in a pipeline after the HashingVectorizer.
            If not ``use_hashing`` then the ``use_idf`` parameter of the TfidfVectorizer will
            be set to use_idf. This step is important because, as explained by the
            `scikit-learn documentation <https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting>`_:
            "In a large text corpus, some words will be very present (e.g. 'the', 'a',
            'is' in English) hence carrying very little meaningful information about the
            actual contents of the document. If we were to feed the direct count data
            directly to a classifier those very frequent terms would shadow the frequencies
            of rarer yet more interesting terms. In order to re-weight the count features
            into floating point values suitable for usage by a classifier it is very common
            to use the tf–idf transform."
        n_features (int, optional): Specifies the number of features/words to use in the vocabulary (which are 
            the rows of the document-term matrix). In the case of the TfidfVectorizer
            the ``n_features`` acts as a maximum since the max_df and min_df parameters
            choose words to add to the vocabulary (to use as features) that occur within
            the bounds specified by these parameters. This value should probably be lowered
            if ``use_hasing`` is set to True. Defaults to 10000.
        lsa_num_components (int, optional): If set then preprocess the data using latent semantic analysis to
            reduce the dimensionality to ``lsa_num_components`` components. Defaults to False.
    
    Returns:
        [list or tuple]: list of features extracted and optionally the u, sigma, and v of the svd calculation on the document-term matrix. only returns if ``return_lsa_svd`` set to True.
    """    
    logger.debug("Extracting features using a sparse vectorizer")
    t0 = time()
    if use_hashing:
        if use_idf:
            # Perform an IDF normalization on the output of HashingVectorizer
            hasher = HashingVectorizer(n_features=n_features,
                                       stop_words='english', alternate_sign=False,
                                       norm=None)
            vectorizer = make_pipeline(hasher, TfidfTransformer())
        else:
            vectorizer = HashingVectorizer(n_features=n_features,
                                           stop_words='english',
                                           alternate_sign=False, norm='l2')
    else:
        vectorizer = TfidfVectorizer(max_df=0.5, max_features=n_features,
                                     min_df=2, stop_words='english',
                                     use_idf=use_idf)
    features = vectorizer.fit_transform(data)
    
    logger.debug("done in %fs" % (time() - t0))
    logger.debug("n_samples: %d, n_features: %d" % features.shape)

    if return_lsa_svd:
        doc_term_matrix = features.toarray()
        doc_term_matrix = doc_term_matrix.transpose(1, 0)
        lsa_svd = np.linalg.svd(doc_term_matrix, full_matrices=False)
        logger.debug("SVD successfully calculated")

    if lsa_num_components:
        logger.debug("Performing dimensionality reduction using LSA")
        t0 = time()
        # Vectorizer results are normalized, which makes KMeans behave as
        # spherical k-means for better results. Since LSA/SVD results are
        # not normalized, we have to redo the normalization.
        svd = TruncatedSVD(lsa_num_components)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)

        features = lsa.fit_transform(features)

        logger.debug("done in %fs" % (time() - t0))

        explained_variance = svd.explained_variance_ratio_.sum()
        logger.debug("Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))
    
    if return_lsa_svd:
        return features, lsa_svd
    else:
        return features

def extract_features_neural_hf(sentences, model="roberta-base", tokenizer="roberta-base", n_hidden=768, squeeze=True, **kwargs):
    """ Extract features using a transformer model from the huggingface/transformers library """
    nlp = pipeline('feature-extraction', model=model, tokenizer=tokenizer, **kwargs)
    features = list()
    vec = np.zeros((len(sentences), n_hidden))
    logger.debug("Extracting features using the " + str(model) + " huggingface neural model")
    for idx, text in tqdm(enumerate(sentences), desc="Extracting Features", total=len(sentences)):
        hidden_state = nlp(text)
        # "mean" averaging approach discussed for beginners at: https://github.com/BramVanroy/bert-for-inference/blob/master/introduction-to-bert.ipynb
        sentence_embedding = np.mean(hidden_state, axis=1)
        if squeeze: # removes the batch dimension
            sentence_embedding = sentence_embedding.squeeze()
        vec[idx] = sentence_embedding
    return vec

def extract_features_neural_sbert(sentences, model="roberta-base-nli-mean-tokens"):
    """ Extract features using Sentence-BERT (SBERT) or SRoBERTa from the sentence-transformers library """
    if model == "roberta":
        model = "roberta-base-nli-mean-tokens"
    elif model == "bert":
        model = "bert-base-nli-mean-tokens"
    nlp = SentenceTransformer(model)
    logger.debug("Extracting features using the sentence level " + str(model) + " model. This is the best method.")
    sentence_embeddings = nlp.encode(sentences)
    return np.array(sentence_embeddings)

def extract_features_spacy(sentences):
    tokens = list()
    logger.debug("Extracting features using spacy. This method cannot tell which spacy model was used but it is highly recommended to use the medium or large model because the small model only includes context-sensitive tensors.")
    for sentence in sentences:
        # https://spacy.io/api/span#vector
        # A real-valued meaning representation. Defaults to an average of the token vectors.
        tokens.append(sentence.vector)
    return np.array(tokens)

def cluster(text, coverage_percentage=0.70, final_sort_by=None, cluster_summarizer="extractive",
            title_generation=False, num_topics=10, minibatch=False, feature_extraction="neural_sbert", **kwargs):
    """Summarize ``text`` to ``coverage_percentage`` length of the original document by extracting features
    from the text, clustering based on those features, and finally summarizing each cluster.
    See the `scikit-learn documentation on clustering text <https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html>`_ 
    for more information since several sections of this function were borrowed from that example.
    
    Notes:
        * ``**kwargs`` is passed to the feature extraction function, which is either :meth:`~summarization_approaches.extract_features_bow` or :meth:`summarization_approaches.extract_features_neural` depending on the ``feature_extraction`` argument.

    Arguments:
        text (str): a string of text to summarize
        coverage_percentage (float, optional): The length of the summary as a percentage of the original document. Defaults to 0.70.
        final_sort_by (str, optional): If `cluster_summarizer` is extractive and `title_generation` is False then
            this argument is available. If specified, it will sort the final cluster
            summaries by the specified string. Options are ``["order", "rating"]``. Defaults to None.
        cluster_summarizer (str, optional): Which summarization method to use to summarize each individual cluster.
            "Extractive" uses the same approach as :meth:`~summarization_approaches.keyword_based_ext` 
            but instead of using keywords from another document, the keywords are
            calculated in the ``TfidfVectorizer`` or ``HashingVectorizer``. Each keyword
            is a feature in the document-term matrix, thus the number of words to use
            is specified by the `n_features` parameter. Options are ``["extractive", "abstractive"].`` 
            Defaults to "extractive".
        title_generation (bool, optional): Option to generate titles for each cluster. Can not be used if
            ``final_sort_by`` is set. Generates titles by summarizing the text using
            BART finetuned on XSum (a dataset of news articles and one sentence
            summaries aka headline generation) and forcing results to be from 1 to
            10 words long. Defaults to False.
        num_topics (int, optional): The number of clusters to create. This should be set to the number of topics
            discussed in the lecture if generating good titles is desired. If separating
            into groups is not very important and a final summary is desired then this
            parameter is not incredibly important, it just should not be set super
            low (3) or super high (50) unless your document in super short or long. Defaults to 10.
        minibatch (bool, optional): Two clustering algorithms are used: ordinary k-means and its more scalable
            cousin minibatch k-means. Setting this to True will use minibatch k-means
            with a batch size set to the number of clusters set in ``num_topics``. Defaults to False.
        feature_extraction (str, optional): Specify how features should be extracted from the text.
            
            * ``neural_hf``: uses a huggingface/transformers pipeline with the roberta model by default
            * ``neural_sbert``: special bert and roberta models fine-tuned to extract sentence embeddings 
                
                * GitHub: https://github.com/UKPLab/sentence-transformers
                * Paper: https://arxiv.org/abs/1908.10084

            * ``spacy``: uses spacy model. All other options use the small spacy model to split
                    the text into sentences since sentence detection does not improve
                    with larger models. However, if spacy is specified for `feature_selection`
                    than the `en_core_web_lg` model will be used to extract high-quality embeddings
            * ``bow``: bow = "bag of words". this method is extremely fast since it is based on
                    word frequencies throughout the input text. The :meth:`~summarization_approaches.extract_features_bow`
                    function contains more details on recommended parameters that you can
                    pass to this function because of ``**kwargs``.
            
            Options are ``["neural_hf", "neural_sbert", "spacy", "bow"]`` Default is "neural_sbert".

    Raises:
        Exception: If incorrect parameters are passed.
    
    Returns:
        [str]: The summarized text as a normal string. Line breaks will be included if ``title_generation`` is true.
    """    
    assert cluster_summarizer in ["extractive", "abstractive"]
    assert feature_extraction in ["neural_hf", "neural_sbert", "spacy", "bow"]
    if (cluster_summarizer == "extractive") and (feature_extraction != "bow"):
        raise Exception("If cluster_summarizer is set to 'extractive', feature_extraction cannot be set to 'bow' because extractive summarization is based off the ranks calculated from the document-term matrix used for 'bow' feature extraction.")
    if final_sort_by:
        assert final_sort_by in ["order", "rating"]
        
        if title_generation: # if final_sort_by and title_generation
            raise Exception("Cannot sort by " + str(final_sort_by) + " and generate titles. Only one option can be specified at a time. In order to generate titles the clusters must not be resorted so each title coresponds to a cluster.")
    
    # If spacy is selected then return the `NLP_SENTENCES` as spacy Span objects instead of strings
    # so they have the `vector` property. Also use the large model to get *real* word vectors.
    # See: https://spacy.io/usage/vectors-similarity
    if feature_extraction == "spacy":
        NLP_DOC, NLP_SENTENCES, NLP_SENTENCES_SPAN, NLP_SENTENCES_LEN, NLP_SENTENCES_LEN_RANGE = get_sentences(text, model="en_core_web_lg")
    else:
        NLP_DOC, NLP_SENTENCES, NLP_SENTENCES_SPAN, NLP_SENTENCES_LEN, NLP_SENTENCES_LEN_RANGE = get_sentences(text)

    if cluster_summarizer == "abstractive":
        NLP_WORDS = [token.text for token in NLP_DOC if token.is_stop != True and token.is_punct != True]
        NLP_WORDS_LEN = len(NLP_WORDS)
        ABS_MIN_LENGTH = int(coverage_percentage * NLP_WORDS_LEN / num_topics)
        logger.debug(str(NLP_WORDS_LEN) + " (Number of Words in Document) * " + str(coverage_percentage) + " (Coverage Percentage) / " + str(num_topics) + " (Number Topics/Clusters) = " + str(ABS_MIN_LENGTH) + " (Abstractive Summary Minimum Length per Cluster)")
    else:
        NUM_SENTENCES_IN_SUMMARY = int(NLP_SENTENCES_LEN * coverage_percentage)
        logger.debug(str(NLP_SENTENCES_LEN) + " (Number of Sentences in Doc) * " + str(coverage_percentage) + " (Coverage Percentage) = " + str(NUM_SENTENCES_IN_SUMMARY) + " (Number of Sentences in Summary)")
        NUM_SENTENCES_PER_CLUSTER = int(NUM_SENTENCES_IN_SUMMARY / num_topics)
        logger.debug(str(NUM_SENTENCES_IN_SUMMARY) + " (Number of Sentences in Summary) / " + str(num_topics) + " (Number Topics/Clusters) = " + str(NUM_SENTENCES_PER_CLUSTER) + " (Number of Sentences per Cluster")
    
    if feature_extraction == "bow":
        if cluster_summarizer == "extractive":
            X, lsa_svd = extract_features_bow(NLP_SENTENCES, return_lsa_svd=True, **kwargs)
            u, sigma, v = lsa_svd
            ranks = compute_ranks(sigma, v)
            logger.debug("Ranks calculated")
        else:
            X = extract_features_bow(NLP_SENTENCES, **kwargs)
    elif feature_extraction == "spacy":
        X = extract_features_spacy(NLP_SENTENCES_SPAN)
    else: # `feature_extraction` contains "neural"
        if "sbert" in feature_extraction:
            X = extract_features_neural_sbert(NLP_SENTENCES, **kwargs)
        else:
            X = extract_features_neural_hf(NLP_SENTENCES, **kwargs)

    if minibatch:
        km = MiniBatchKMeans(n_clusters=num_topics, init_size=1000, batch_size=1000)
    else:
        km = KMeans(n_clusters=num_topics, max_iter=100)

    logger.debug("Clustering data with %s" % km)
    t0 = time()
    km.fit(X)
    logger.debug("done in %0.3fs" % (time() - t0))

    sentence_clusters = [[] for _ in range(num_topics)] # initialize array with `num_topics` empty arrays inside

    if cluster_summarizer == "extractive":
        SentenceInfo = namedtuple("SentenceInfo", ("sentence", "order", "rating",))
        infos = (SentenceInfo(*t) for t in zip(NLP_SENTENCES, NLP_SENTENCES_LEN_RANGE, ranks))
    else:
        SentenceInfo = namedtuple("SentenceInfo", ("sentence", "order",))
        infos = (SentenceInfo(*t) for t in zip(NLP_SENTENCES, NLP_SENTENCES_LEN_RANGE))
    logger.debug("Created sentence info tuples")

    # Add sentence info tuples to the list representing their cluster number.
    # If a sentence belongs to cluster 3, it is added to list 3 of the sentence_clusters master list.
    for info in infos:
        cluster_num = km.labels_[info.order]
        sentence_clusters[cluster_num].append(info)
    logger.debug("Sorted info tuples by cluster")
    
    if title_generation:
        final_sentences = list()
    else:
        final_sentences = ""
    titles = list()

    if cluster_summarizer == "abstractive":
        summarizer_content = initialize_abstractive_model("bart")
    if title_generation:
        summarizer_title = initialize_abstractive_model("facebook/bart-large-xsum")

    for idx, cluster in tqdm(enumerate(sentence_clusters), desc="Summarizing Clusters", total=len(sentence_clusters)):
        if cluster_summarizer == "extractive":
            # If `title_generation` is enabled then create a single string holding the unsummarized 
            # sentences so it can be passed to the title generation algorithm. Also, if 
            # `title_generation` is enabled then 
            if title_generation:
                cluster_unsummarized_sentences = " ".join([i.sentence for i in cluster])
            
            # sort sentences by rating in descending order
            cluster = sorted(cluster, key=attrgetter("rating"), reverse=True)
            # get `count` first best rated sentences
            cluster = cluster[:NUM_SENTENCES_PER_CLUSTER]
            # sort sentences by their order in document
            cluster = sorted(cluster, key=attrgetter("order"))

            # if `title_generation` is enabled then the final cluster should be a string of sentences
            if title_generation:
                cluster = " ".join([i.sentence for i in cluster])

        else:
            # sort sentences by their order in document
            cluster = sorted(cluster, key=attrgetter("order"))
            # combine sentences in cluster into string
            cluster_unsummarized_sentences = " ".join([i.sentence for i in cluster])
            # summarize the sentences
            cluster = generic_abstractive(cluster_unsummarized_sentences, summarizer_content, min_length=ABS_MIN_LENGTH)

        if title_generation:
            final_sentences.append(cluster)
        else:
            final_sentences += cluster

        if title_generation:
            # generate a title by running the 
            title = generic_abstractive(cluster_unsummarized_sentences, summarizer_title,
                                        min_length=1, max_length=10)
            titles.append(title)

    if cluster_summarizer == "extractive":
        if final_sort_by and not title_generation:
            final_sentences = sorted(final_sentences, key=attrgetter(final_sort_by))
            logger.debug("Extractive - Final sentences sorted by " + str(final_sort_by))
        else:
            # sort by cluster is default
            logger.debug("Extractive - Final sentences sorted by cluster")
        if not title_generation: # if extractive and not generating titles
            final_sentences = " ".join([i.sentence for i in final_sentences])
    
    if title_generation:
        final = ""
        for idx, group in enumerate(final_sentences):
            final += ("Title: " + titles[idx] + "\nContent: " + group + "\n\n")
        return final

    return final_sentences
    
def initialize_abstractive_model(sum_model, use_hf_pipeline=True, *args, **kwargs):
    logger.debug("Loading " + sum_model + " model")
    if use_hf_pipeline:
        if sum_model == "bart":
            sum_model = "sshleifer/distilbart-cnn-12-6"
        SUMMARIZER = pipeline("summarization", model=sum_model, tokenizer="facebook/bart-large-cnn")
    else:
        if sum_model == "bart":
            import bart_sum
            SUMMARIZER = bart_sum.BartSumSummarizer(*args, **kwargs)
        elif sum_model == "presumm":
            import presumm.presumm as presumm
            SUMMARIZER = presumm.PreSummSummarizer(*args, **kwargs)
        else:
            logger.error("Valid model was not specified in `sum_model`. Returning -1.")
            return -1
    logger.debug(sum_model + " model loaded successfully")
    return SUMMARIZER

def generic_abstractive(to_summarize, summarizer=None, min_length=None, max_length=None, *args, **kwargs):
    if isinstance(summarizer, str):
        summarizer = initialize_abstractive_model(summarizer, *args, **kwargs)
    if not summarizer:
        summarizer = initialize_abstractive_model("bart")

    if not min_length:
        TO_SUMMARIZE_LENGTH = len(to_summarize.split())
        min_length = int(TO_SUMMARIZE_LENGTH/6)
        if min_length > 512:
            # If the length is too long the model will start to repeat
            min_length = 512
    if not max_length:
        max_length = min_length+200
    LECTURE_SUMMARIZED = summarizer(to_summarize, min_length=min_length, max_length=max_length)
    
    if type(LECTURE_SUMMARIZED) is list:  # hf pipeline was used
        return LECTURE_SUMMARIZED[0]["summary_text"]

    return LECTURE_SUMMARIZED

def create_sumy_summarizer(algorithm, language="english"):
    stemmer = Stemmer(language)

    if algorithm == "lsa":
        summarizer = LsaSummarizer(stemmer)
    elif algorithm == "luhn":
        summarizer = LuhnSummarizer(stemmer)
    elif algorithm == "lex_rank":
        summarizer = LexRankSummarizer(stemmer)
    elif algorithm == "text_rank":
        summarizer = TextRankSummarizer(stemmer)
    elif algorithm == "edmundson":
        summarizer = EdmundsonSummarizer(stemmer)
    elif algorithm == "random":
        summarizer = RandomSummarizer(stemmer)

    return summarizer

def generic_extractive_sumy(text, coverage_percentage=0.70, algorithm="text_rank", language="english"):
    _, _, _, NLP_SENTENCES_LEN, _ = get_sentences(text)

    # text = " ".join([token.text for token in NLP_DOC if token.is_stop != True])

    NUM_SENTENCES_IN_SUMMARY = int(NLP_SENTENCES_LEN * coverage_percentage)
    logger.debug(str(NLP_SENTENCES_LEN) + " (Number of Sentences in Doc) * " + str(coverage_percentage) + " (Coverage Percentage) = " + str(NUM_SENTENCES_IN_SUMMARY) + " (Number of Sentences in Summary)")

    parser = PlaintextParser.from_string(text, Tokenizer(language))

    summarizer = create_sumy_summarizer(algorithm, language)
    logger.debug("Sumy Summarizer initialized successfully")

    summarizer.stop_words = get_stop_words(language)

    sentence_list = [str(sentence) for sentence in summarizer(parser.document, NUM_SENTENCES_IN_SUMMARY)]

    return " ".join(sentence_list)

def structured_joined_sum(ssa_path, transcript_json_path, frame_every_x=1, ending_char=" ", to_json=False, summarization_method="abstractive", *args, **kwargs):
    """Summarize slides by combining the Slide Structure Analysis (SSA) and transcript json 
    to create a per slide summary of the transcript. The content from the beginning of one 
    slide to the start of the next to the nearest ``ending_char`` is considered the transcript 
    that belongs to that slide. The summarized transcript content is organzed in a dictionary 
    where the slide titles are keys. This dictionary can be returned as json or written to a 
    json file.

    Args:
        ssa_path (str): Path to the SSA JSON file.
        transcript_json_path (str): Path to the transcript JSON file.
        frame_every_x (int, optional): How often frames were extracted from the video that the SSA
            was conducted on. This is used to convert frame numbers to time (seconds). Defaults to 1.
        ending_char (str, optional): The character that the transcript belonging to each slide will
            be extended to. For instance, if the next slide appears in the middle of a word, the 
            transcript content will continue to be added to the previous slide until the 
            ``ending_char`` is reached. It is recommended to use  periods or a special end of 
            sentence token if present. These can be generated with 
            :meth:`transcribe.transcribe_main.segment_sentences` Defaults to ``" "`` (nearest 
            complete word).
        to_json (bool or str, optional): If the output dictionary should be returned as a JSON string. 
            This can also be set to a path as a string and the JSON data will be dumped to the file 
            at that path. Defaults to False.
        summarization_method (str, optional): The method to use to summarize each slide's 
            transcript content. Options include "abstractive", "extractive", or "none". Defaults 
            to "abstractive".
        ``*args`` and ``**kwargs`` are passed to the summarization function, which is either
            :meth:`~summarization_approaches.generic_abstractive` or 
            :meth:`~summarization_approaches.generic_extractive_sumy` depending on 
            ``summarization_method``.

    Returns:
        dict or str: A dictionary containing the slide titles as keys and the summarized transcript 
        content for each slide as values. A string will be returned when ``to_json`` is set. If 
        ``to_json`` is ``True`` (boolean) the JSON data formatted as a string will be returned. 
        If ``to_json`` is a path (string), then the JSON data will be dumped to the file specified
        and the path to the file will be returned.
    """
    assert summarization_method in ["abstractive", "extractive", "none"], "Invalid summarization method"
    
    with open(ssa_path, "r") as ssa_file, open(transcript_json_path, "r") as transcript_json_file:
        ssa = json.load(ssa_file)
        transcript_json = json.load(transcript_json_file)
    
    transcript_json_idx = 0
    current_time = 0

    first_slide_timestamp_seconds = ssa[0]["frame_number"] * frame_every_x
    transcript_before_slides = ""
    while True:
        current_letter_obj = transcript_json[transcript_json_idx]
        if current_time >= first_slide_timestamp_seconds and current_letter_obj["text"] == ending_char:
            break
        try:
            current_time = current_letter_obj["start_time"]
        except KeyError: # no `start_time` so use the previous value
            pass
        transcript_before_slides += current_letter_obj["text"]
        transcript_json_idx += 1

    final_dict = OrderedDict({"Preface": {"transcript": transcript_before_slides}})

    go_to_end = False
    for idx, slide in tqdm(enumerate(ssa), total=len(ssa), desc="Grouping Slides and Transcript"):
        title_lines = [i for i, x in slide["category"].items() if x == 2]

        all_slide_content = []
        current_par_num = 0
        prev_line_num = 0
        for line_idx, line in slide["text"].items():
            stored_line_num = slide["line_num"][line_idx]
            # If the line number did not increase by 1 then assume a new paragraph
            if stored_line_num != prev_line_num + 1:
                current_par_num += 1
            
            # If the line is not footer text and is not a title
            if slide["category"][line_idx] not in (-1, 2):
                # If the line is bold then add "**" on either side
                if slide["category"][line_idx] == 1:
                    line = "**" + line + "**"
                try:
                    # If the previously added line was bold (it ended in "**")
                    if all_slide_content[current_par_num][-2:] == "**":
                        # Remove the first "**" from the line to be added
                        line = line[2:]
                        # Remove the last "**" from the previously added line
                        all_slide_content[current_par_num] = all_slide_content[current_par_num][:-2]
                    # Add the line to the current paragraph with a space to avoid combined words
                    all_slide_content[current_par_num] += (" " + line)
                except IndexError:  # Paragraph not yet created, so create it
                    # Add the first line of a paragraph to `all_slide_content`
                    all_slide_content.append(line)
                
                # Update the previous line number to the current line number
                prev_line_num = stored_line_num
        
        # Only include paragraphs greater than 3 characters
        all_slide_content = [x for x in all_slide_content if len(x) > 3]

        title = " ".join([slide["text"][line] for line in title_lines]).strip()
        if not title:
            title = "Slide {}".format(idx + 1)
        
        try:
            ssa[idx + 1]
            next_slide_timestamp_seconds = ssa[idx + 1]["frame_number"] * frame_every_x
        except IndexError:  # Last item
            go_to_end = True

        coresponding_transcript_text = ""
        while True:
            # If `transcript_json` out of items break the loop
            if transcript_json_idx == len(transcript_json):
                break
            
            current_letter_obj = transcript_json[transcript_json_idx]

            # If `go_to_end` enabled then never break using this statement.
            # If the current time is past the next slide time advance to the next slide.
            # However, jump forward a few letter if necessary in order to end the current
            # transcript-slide segment with `endding_char`.
            if (not go_to_end) and (current_time >= next_slide_timestamp_seconds and current_letter_obj["text"] == ending_char):
                break    

            try:
                current_time = current_letter_obj["start_time"]
            except KeyError: # no `start_time` so use the previous value
                pass

            coresponding_transcript_text += current_letter_obj["text"]
            transcript_json_idx += 1
        
        final_dict[title] = {"transcript": coresponding_transcript_text.strip()}
        final_dict[title]["slide_content"] = all_slide_content
        if "figure_paths" in slide.keys():
            final_dict[title]["figure_paths"] = slide["figure_paths"]

    if summarization_method not in ("none", None):
        if summarization_method == "abstractive":
            summarizer = initialize_abstractive_model("bart")

        for title, content in tqdm(final_dict.items(), desc="Summarizing Slides"):
            content = content["transcript"]
            if summarization_method == "abstractive":
                final_dict[title]["transcript"] = generic_abstractive(content, summarizer, *args, **kwargs)
            else:
                final_dict[title]["transcript"] = generic_extractive_sumy(content, *args, **kwargs)

    if to_json:
        if type(to_json) is bool:
            return json.dumps(final_dict)
        else:
            with open(to_json, "w+") as json_file:
                json.dump(final_dict, json_file)
                return to_json
    
    return final_dict


# structured_joined_sum("remove2.json", "process/audio.json")
