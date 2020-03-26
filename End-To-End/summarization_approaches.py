import logging
import sys
import os
import spacy
import math
from time import time
from tqdm import tqdm
from collections import namedtuple, defaultdict
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

# sys.path.insert(1, os.path.join(sys.path[0], '../Models/slide-classifier'))
# from class_cluster_scikit import Cluster #pylint: disable=import-error,wrong-import-position

sys.path.insert(1, os.path.join(sys.path[0], '../Models/summarizer'))

logger = logging.getLogger(__name__)

def get_complete_sentences(text, return_string=False):
    nlp = spacy.load("en_core_web_lg")
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

def get_sentences(text):
    logger.debug("Tokenizing text...")
    nlp = spacy.load("en_core_web_lg")
    NLP_DOC = nlp(text)
    logger.debug("Text tokenized successfully")
    NLP_SENTENCES = [str(sentence) for sentence in list(NLP_DOC.sents)]
    NLP_SENTENCES_LEN = len(NLP_SENTENCES)
    NLP_SENTENCES_LEN_RANGE = range(NLP_SENTENCES_LEN)

    return NLP_DOC, NLP_SENTENCES, NLP_SENTENCES_LEN, NLP_SENTENCES_LEN_RANGE

def keyword_based_ext(ocr_text, transcript_text, coverage_percentage=0.70):
    from summa import keywords
    ocr_text = ocr_text.replace('\n', ' ').replace('\r', '')

    ocr_keywords = keywords.keywords(ocr_text)
    ocr_keywords = ocr_keywords.splitlines()
    logger.debug("Number of keywords: " + str(len(ocr_keywords)))

    vocab = dict(zip(ocr_keywords, range(0, len(ocr_keywords))))

    vectorizer = TfidfVectorizer(vocabulary=vocab, stop_words="english")

    _, NLP_SENTENCES, NLP_SENTENCES_LEN, _ = get_sentences(text)

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

def cluster(text, coverage_percentage=0.70, final_sort_by=None, cluster_summarizer="extractive",
            title_generation=False, num_topics=10, use_hashing=False, use_idf=True, n_features=10000,
            lsa_num_components=False, minibatch=False):
    """Summarize `text` to `coverage_percentage` length of the original document by extracting features
    from the text, clustering based on those features, and finally summarizing each cluster.
    See the scikit-learn documentation on clustering text for more information since several chunks
    of this function were borrowed from that example: https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html
    
    Arguments:
        text {str} -- a string of text to summarize
    
    Keyword Arguments:
        coverage_percentage {float} -- The length of the summary as a percentage of the original document. (default: {0.70})
        final_sort_by {str} -- If `cluster_summarizer` is extractive and `title_generation` is False then
                               this argument is available. If specified, it will sort the final cluster
                               summaries by the specified string. (options: {"order", "rating"}) (default: {None})
        cluster_summarizer {str} -- Which summarization method to use to summarize each individual cluster.
                                    "Extractive" uses the same approach as the keyword_based_ext() function
                                    but instead of using keywords from another document, the keywords are
                                    calculated in the TfidfVectorizer or HashingVectorizer. Each keyword
                                    is a feature in the document-term matrix, thus the number of words to use
                                    is specified by the `n_features` parameter. (options: {"extractive", "abstractive"}) (default: {"extractive"})
        title_generation {bool} -- Option to generate titles for each cluster. Can not be used if
                                   `final_sort_by` is set. Generates titles by summarizing the text using
                                   BART finetuned on XSum (a dataset of news articles and one sentence
                                   summaries aka headline generation) and forcing results to be from 1 to
                                   10 words long. (default: {False})
        num_topics {int} -- The number of clusters to create. This should be set to the number of topics
                            discussed in the lecture if generating good titles is desired. If separating
                            into groups is not very important and a final summary is desired then this
                            parameter is not incredibly important, it just should not be set super
                            low (3) or super high (50) unless your document in super short or long. (default: {10})
        use_hashing {bool} -- Use a HashingVectorizer instead of a CountVectorizer. (default: {False})
                              A HashingVectorizer should only be used with large datasets. Large to the
                              degree that you'll probably never pass enough data through this function
                              to warrent the usage of a HashingVectorizer. HashingVectorizers use very
                              little memory and are thus scalable to large datasets because there is no 
                              need to store a vocabulary dictionary in memory.
                              More information can be found in the scikit-learn documentation:
                              https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html
        use_idf {bool} -- Option to use inverse document-frequency. In the case of `use_hasing`
                          a TfidfTransformer will be appended in a pipeline after the HashingVectorizer.
                          If not `use_hashing` then the `use_idf` parameter of the TfidfVectorizer will
                          be set to use_idf. This step is important because, as explained by the
                          scikit-learn documentation:
                          "In a large text corpus, some words will be very present (e.g. "the", "a",
                          "is" in English) hence carrying very little meaningful information about the
                          actual contents of the document. If we were to feed the direct count data
                          directly to a classifier those very frequent terms would shadow the frequencies
                          of rarer yet more interesting terms. In order to re-weight the count features
                          into floating point values suitable for usage by a classifier it is very common
                          to use the tf–idf transform."
                          More info: https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting (default: {True})
        n_features {int} -- Specifies the number of features/words to use in the vocabulary (which are 
                            the rows of the document-term matrix). In the case of the TfidfVectorizer
                            the `n_features` acts as a maximum since the max_df and min_df parameters
                            choose words to add to the vocabulary (to use as features) that occur within
                            the bounds specified by these parameters. This value should probably be lowered
                            if `use_hasing` is set to True. (default: {10000})
        lsa_num_components {int} -- If set then preprocess the data using latent semantic analysis to
                                    reduce the dimensionality to `lsa_num_components` components. (default: {False})
        minibatch {bool} -- Two clusterign algorithms are used: ordinary k-means and its more scalable
                            cousin minibatch k-means. Setting this to True will use minibatch k-means
                            with a batch size set to the number of clusters set in `num_topics`. (default: {False})
    
    Raises:
        Exception: If incorrect parameters are passed.
    
    Returns:
        [str] -- The summarized text as a normal string. Line breaks will be included if
        `title_generation` is true
    """    
    assert cluster_summarizer in ["extractive", "abstractive"]
    if final_sort_by:
        assert final_sort_by in ["order", "rating"]
        
        if title_generation: # if final_sort_by and title_generation
            raise Exception("Cannot sort by " + str(final_sort_by) + " and generate titles. Only one option can be specified at a time. In order to generate titles the clusters must not be resorted so each title coresponds to a cluster.")
    
    vectorizer = TfidfVectorizer(stop_words="english")

    _, NLP_SENTENCES, NLP_SENTENCES_LEN, NLP_SENTENCES_LEN_RANGE = get_sentences(text)

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
    X = vectorizer.fit_transform(NLP_SENTENCES)
    
    logger.debug("done in %fs" % (time() - t0))
    logger.debug("n_samples: %d, n_features: %d" % X.shape)

    if cluster_summarizer == "extractive":
        doc_term_matrix = X.toarray()
        doc_term_matrix = doc_term_matrix.transpose(1, 0)
        u, sigma, v = np.linalg.svd(doc_term_matrix, full_matrices=False)
        logger.debug("SVD successfully calculated")

        ranks = compute_ranks(sigma, v)
        logger.debug("Ranks calculated")

    if lsa_num_components:
        logger.debug("Performing dimensionality reduction using LSA")
        t0 = time()
        # Vectorizer results are normalized, which makes KMeans behave as
        # spherical k-means for better results. Since LSA/SVD results are
        # not normalized, we have to redo the normalization.
        svd = TruncatedSVD(lsa_num_components)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)

        X = lsa.fit_transform(X)

        logger.debug("done in %fs" % (time() - t0))

        explained_variance = svd.explained_variance_ratio_.sum()
        logger.debug("Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))

    if minibatch:
        km = MiniBatchKMeans(n_clusters=num_topics, init_size=1000, batch_size=1000)
    else:
        km = KMeans(n_clusters=num_topics, max_iter=100)

    logger.debug("Clustering sparse data with %s" % km)
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
        summarizer_title = initialize_abstractive_model("bart", pretrained="bart.large.xsum", hg_transformers=False)

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
                                        min_length=1, max_length_b=10)
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
    
def initialize_abstractive_model(sum_model, *args, **kwargs):
    logger.debug("Loading " + sum_model + " model")
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

def generic_abstractive(to_summarize, summarizer=None, min_length=None, max_length_b=None, *args, **kwargs):
    if isinstance(summarizer, str):
        summarizer = initialize_abstractive_model(summarizer, *args, **kwargs)
    if not summarizer:
        summarizer = initialize_abstractive_model("bart")

    if not min_length:
        TO_SUMMARIZE_LENGTH = len(to_summarize.split())
        min_length = int(TO_SUMMARIZE_LENGTH/6)
        if min_length > 500:
            # If the length is too long the model will start to repeat
            min_length = 500
    if not max_length_b:
        max_length_b = min_length+200
    LECTURE_SUMMARIZED = summarizer.summarize_string(to_summarize, min_len=min_length, max_len_b=max_length_b)

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
    _, _, NLP_SENTENCES_LEN, _ = get_sentences(text)

    # text = " ".join([token.text for token in NLP_DOC if token.is_stop != True])

    NUM_SENTENCES_IN_SUMMARY = int(NLP_SENTENCES_LEN * coverage_percentage)
    logger.debug(str(NLP_SENTENCES_LEN) + " (Number of Sentences in Doc) * " + str(coverage_percentage) + " (Coverage Percentage) = " + str(NUM_SENTENCES_IN_SUMMARY) + " (Number of Sentences in Summary)")

    parser = PlaintextParser.from_string(text, Tokenizer(language))

    summarizer = create_sumy_summarizer(algorithm, language)
    logger.debug("Sumy Summarizer initialized successfully")

    summarizer.stop_words = get_stop_words(language)

    sentence_list = [str(sentence) for sentence in summarizer(parser.document, NUM_SENTENCES_IN_SUMMARY)]

    return " ".join(sentence_list)