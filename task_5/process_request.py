import codecs
import os
from argparse import ArgumentParser
from collections import Counter
from typing import Dict

from natasha import Segmenter, NewsMorphTagger, MorphVocab, NewsEmbedding
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from task_2.code.task_2 import get_lemmatized_doc
from task_3.utils import load_dict
from task_5.utils import load_tf_idf_matrix_from_file, load_vocab_idfs


def vectorize_request_tf_idf(request_raw_text: str, segmenter: Segmenter, morph_tagger: NewsMorphTagger,
                             morph_vocab: MorphVocab, token2id: Dict[str, int],
                             token_idfs: Dict[int, float]) -> csr_matrix:
    """
    :param request_raw_text: Непредобработанная строка поискового запроса
    :param segmenter: Токенизатор библиотеки Natasha
    :param morph_tagger: Морфологический парсер библиотеки Natasha. Морфологический
    разбор необходим для лемматизации: лемматизатор библиотеки Natasha использует
    его для при лемматизации
    :param morph_vocab: Лемматизатор библиотеки Natasha
    :param token2id: Словарь: маппинг из термина в идентификатор слова в словаре
    :param token_idfs: Словарь: маппинг из номера термина в словаре в его значение IDF
    :return: Разреженный TF-IDF вектор запроса
    """
    vocab_size = len(token_idfs.keys())
    # Токенизируем и лемматизируем документ
    lemmatized_tokens = get_lemmatized_doc(raw_text=request_raw_text, segmenter=segmenter,
                                           morph_tagger=morph_tagger, morph_vocab=morph_vocab)

    # Превращаем список слов в список номеров слов в словаре
    token_ids_list = [token2id[token] for token in lemmatized_tokens if token in token2id.keys()]
    # Подсчитываем частоты слов в документе
    request_tf = Counter(token_ids_list)
    tf_idf_vector_col_indices = []
    tf_idf_vector_values = []
    # Подсчитываем TF-IDF каждого слова в документе
    for token_id, token_tf in request_tf.items():
        tf_idf_vector_col_indices.append(token_id)
        token_idf = token_idfs[token_id]
        tf_idf_vector_values.append(token_tf * token_idf)
    tf_idf_vector_row_indices = [0] * len(tf_idf_vector_col_indices)
    assert len(tf_idf_vector_col_indices) == len(tf_idf_vector_values)
    # Создаём TF-IDF вектор документа как разреженный вектор
    request_sparse_tf_idf_vector = csr_matrix(
        (tf_idf_vector_values, (tf_idf_vector_row_indices, tf_idf_vector_col_indices)),
        shape=(1, vocab_size))
    return request_sparse_tf_idf_vector


def write_request_log(log_file_path: str, request_str: str, response_document_id: int, response_document_path: str):
    """
    Логирует результат выполнения поискового запроса
    :param log_file_path: Путь к лог-файлу
    :param request_str: Строка запроса
    :param response_document_id: Идентификатор документа - ответа на запрос
    :param response_document_path: Путь до файла текста исходного документа. Его содержимое также логируется,
    обрамлённое строками '---'
    """
    with codecs.open(log_file_path, 'a+', encoding="utf-8") as log_file, \
            codecs.open(response_document_path, 'r', encoding="utf-8") as raw_text_file:
        document_text = raw_text_file.read()
        log_file.write(f"Строка запроса: {request_str}\n")
        log_file.write(f"Номер документа - ответа на запрос: {response_document_id}\n")
        log_file.write(f"Текст документа:\n---\n{document_text.strip()}\n---\n\n")


def main():
    parser = ArgumentParser()
    parser.add_argument('--input_request_str', default=r"Классическая литература", type=str,
                        help="Строка поискового запроса")
    parser.add_argument('--input_dict_path', default=r"../task_2/tokenized_texts/dict.txt", type=str,
                        help="Путь к словарю")
    parser.add_argument('--input_df_path', default="../task_4/tf_idf/df.txt",
                        type=str, help=r"Путь до файла с документными частотами терминов."
                                       r"Записываю DF в файл, потому что почему бы и нет, так нагляднее")
    parser.add_argument('--input_tf_idf_path', default="../task_4/tf_idf/tf_idf.txt",
                        type=str, help=r"Путь до файла cо значениями TF-IDF. Каждая строка соответствует одному"
                                       r"документу. В строке пробелами разделены пары <термин, его idf, его tf-idf>,"
                                       r"а термин и его tf-idf разделены строкой '~~~'")
    parser.add_argument('--input_raw_documents_dir', default=r"../task_1/reviews/reviews/", type=str,
                        help="Путь к директории непредобработанных документов")
    parser.add_argument('--output_log_path', default=r"search_log.txt", type=str,
                        help="Путь к файлу логов поисковых запросов")
    args = parser.parse_args()
    input_request_str = args.input_request_str
    input_dict_path = args.input_dict_path
    input_df_path = args.input_df_path
    input_tf_idf_path = args.input_tf_idf_path
    input_raw_documents_dir = args.input_raw_documents_dir
    output_log_path = args.output_log_path
    output_dir = os.path.dirname(output_log_path)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)

    # Подгружаем словарь в память
    token2id = load_dict(input_dict_path)
    # Подгружаем предпосчитанную матрицу TF-IDF из файла
    tf_idf_matrix = load_tf_idf_matrix_from_file(tf_idf_file_path=input_tf_idf_path, token2id=token2id, )
    num_documents = tf_idf_matrix.shape[0]
    # Подгружаем инвертированные документные частоты терминов (IDF)
    token_idfs = load_vocab_idfs(vocab_dfs_path=input_df_path, token2id=token2id, num_documents=num_documents)

    segmenter = Segmenter()
    morph_vocab = MorphVocab()
    emb = NewsEmbedding()
    morph_tagger = NewsMorphTagger(emb)
    request_vector = vectorize_request_tf_idf(request_raw_text=input_request_str, segmenter=segmenter,
                                              morph_tagger=morph_tagger, morph_vocab=morph_vocab, token2id=token2id,
                                              token_idfs=token_idfs)
    # Идентификатор документа, наиболее похожего на запрос векторно. Мера похожести - косинусная близость векторов
    response_document_id = cosine_similarity(tf_idf_matrix, request_vector).argmax()
    # Путь до файла исходного непредобработанного документа
    response_document_path = os.path.join(input_raw_documents_dir, f"review_{response_document_id}.txt")
    # Логируем результат выполнения запроса
    write_request_log(log_file_path=output_log_path, request_str=input_request_str,
                      response_document_id=response_document_id, response_document_path=response_document_path)


if __name__ == '__main__':
    main()
