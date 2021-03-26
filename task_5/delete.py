import math
from collections import Counter

from natasha import Segmenter, NewsMorphTagger, MorphVocab

from task_2.code.task_2 import get_lemmatized_doc


def vectorize_request_tf_idf(request_raw_text, segmenter: Segmenter, morph_tagger: NewsMorphTagger,
                             morph_vocab: MorphVocab, token2id):
    lemmatized_tokens = get_lemmatized_doc(raw_text=request_raw_text, segmenter=segmenter,
                                           morph_tagger=morph_tagger, morph_vocab=morph_vocab)
    token_ids_list = [token2id[token] for token in lemmatized_tokens]
    unique_document_tokens_ids = set(token_ids_list)
    documents_frequencies.update(unique_document_tokens_ids)
    token_id_frequencies = Counter(token_ids_list)
    # TODO  Берём частоту термина в документе из матрицы TF
    # TODO
    tf = term_frequencies_sparse_matrix[doc_id, token_id]
    #  TODO Считаем инвертированную документную частоту термина (IDF)
    idf = num_documents / documents_frequencies[token_id]
    #  TODO TF-IDF = TF * log(IDF)
    tf_log_idf_value = tf * math.log2(idf)
