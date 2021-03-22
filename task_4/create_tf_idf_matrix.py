import codecs
import math
import os
from argparse import ArgumentParser
from collections import Counter
from typing import Dict, Tuple

from scipy.sparse import csr_matrix

from task_3.utils import load_dict


def get_df_sparse_tf_matrices_from_file(documents_path: str, token2id: Dict[str, int]) \
        -> Tuple[csr_matrix, Counter]:
    """
    Считает матрицу документной частоты и частот терминов (последнюю - в разреженном виде)
    на основе файла с текстами документов.
    :param documents_path: Путь к файлу с текстами документов. Каждая строка содержит слова в точности
    1 документа, разделенные пробелами
    :param token2id: Словарь: маппинг из термина в идентификатор слова в словаре
    :return: term_frequencies_sparse_matrix - разреженная TF матрица: матрица размера
    (число документов, размер словаря), содержащая частоты слов в документах;
    term_frequencies_sparse_matrix - Словарь (вектор) документных частот терминов
    Размера (размер словаря). Значения - число документов, в которых содержится соответствующий термин
    """
    vocab_size = len(token2id.keys())
    documents_frequencies = Counter()
    row_indices = []
    col_indices = []
    frequency_values = []
    num_documents = 0
    with codecs.open(documents_path, 'r', encoding="utf-8") as docs_file:
        for doc_id, line in enumerate(docs_file):
            tokens_ids_list = [token2id[token] for token in line.strip().split()]
            unique_document_tokens_ids = set(tokens_ids_list)
            documents_frequencies.update(unique_document_tokens_ids)
            token_id_frequencies = Counter(tokens_ids_list)
            for token_id, freq in token_id_frequencies.items():
                row_indices.append(doc_id)
                col_indices.append(token_id)
                frequency_values.append(freq)
            num_documents += 1
    term_frequencies_sparse_matrix = csr_matrix((frequency_values, (row_indices, col_indices)),
                                                shape=(num_documents, vocab_size))
    return term_frequencies_sparse_matrix, documents_frequencies


def calculate_sparse_tf_idf_matrix(term_frequencies_sparse_matrix: csr_matrix,
                                   documents_frequencies: Dict[str, int]) -> csr_matrix:
    """
    Считает TF-IDF матрицу на основе матрицы TF и ветора DF
    :param term_frequencies_sparse_matrix: разреженная TF матрица: матрица размера
    (число документов, размер словаря), содержащая частоты слов в документах
    :param documents_frequencies: Словарь (вектор) документных частот терминов
    :return: Разреженная TF-IDF матрица
    """
    # Узнаём общее число документов и размер словаря
    num_documents, vocab_size = term_frequencies_sparse_matrix.shape
    # Создаём пустую разреженную матрицу
    tf_idf_sparse_matrix = csr_matrix((num_documents, vocab_size), dtype=float)
    non_empty_row_ids, non_empty_col_ids = term_frequencies_sparse_matrix.nonzero()
    # Итерируемся по ненулевым элементам разреженной матрицы TF
    for doc_id, token_id in zip(non_empty_row_ids, non_empty_col_ids):
        # Берём частоту термина в документе из матрицы TF
        tf = term_frequencies_sparse_matrix[doc_id, token_id]
        # Считаем инвертированную документную частоту термина (IDF)
        idf = num_documents / documents_frequencies[token_id]
        # TF-IDF = TF * log(IDF)
        tf_log_idf_value = tf * math.log2(idf)
        tf_idf_sparse_matrix[doc_id, token_id] = tf_log_idf_value
    return tf_idf_sparse_matrix


def save_df_matrix(save_path: str, df: Counter, id2token: Dict[int, str]):
    """
    Сохраняет вектор DF в файл. 1 строка=<токен>\t<его частота> в порядке убывания
    документной частоты
    :param save_path: Путь к файлу, в который будет сохранен вектор DF
    :param df: Вектор DF
    :param id2token: Инвертированный словарь, возвращающий токен по номеру
    его позиции в словаре
    """
    with codecs.open(save_path, 'w+', encoding="utf-8") as output_file:
        for token_id, frequency in df.most_common():
            output_file.write(f"{id2token[token_id]}\t{frequency}\n")


def save_tf_idf_matrix(save_path: str, tf_idf_sparse_matrix, id2token: Dict[int, str], sep="~~~"):
    """
    Сохраняет TF-IDF матрицу в файл. 1 строка соответствует одному документу.
    Пары <термин, tf-idf термина> разделены между собой пробелом. Термин с его значением
    TF-IDF в каждой паре разделен разделителем sep
    :param save_path: Путь к файлу, в который будут сохранены значения TF-IDF
    :param tf_idf_sparse_matrix: Разреженная матрица TF-IDF
    :param id2token: Инвертированный словарь, возвращающий токен по номеру
    его позиции в словаре
    :param sep: Разделитель между термином и его значением TF-IDF
    """
    current_doc_id = 0
    non_empty_row_ids, non_empty_col_ids = tf_idf_sparse_matrix.nonzero()
    with codecs.open(save_path, 'w+', encoding="utf-8") as output_file:
        for doc_id, token_id in zip(non_empty_row_ids, non_empty_col_ids):
            tf_idf_value = tf_idf_sparse_matrix[doc_id, token_id]
            output_file.write(f"{id2token[token_id]}{sep}{tf_idf_value} ")
            if doc_id != current_doc_id:
                output_file.write("\n")
                current_doc_id = doc_id


def main():
    parser = ArgumentParser()
    parser.add_argument('--input_documents_path', default=r"../task_2/tokenized_texts/documents.txt", type=str,
                        help="Путь к файлу с лемматизированными документами, по 1 строке файла"
                             "на документу")
    parser.add_argument('--input_dict_path', default=r"../task_2/tokenized_texts/dict.txt", type=str,
                        help="Путь к словарю")
    parser.add_argument('--output_df_path', default="tf_idf/df.txt",
                        type=str, help=r"Выходной путь до файла с документными частотами терминов."
                                       r"Записываю DF в файл, потому что почему бы и нет, так нагляднее")
    parser.add_argument('--output_tf_idf_path', default="tf_idf/tf_idf.txt",
                        type=str, help=r"Выходной путь до файла cо значениями TF-IDF. Каждая строка соответствует"
                                       r"одному документу. В строке пробелами разделены пары <термин, tf-idf термина>,"
                                       r"а термин и его tf-idf разделены строкой '~~~'")
    args = parser.parse_args()
    input_documents_path = args.input_documents_path
    input_dict_path = args.input_dict_path
    output_df_path = args.output_df_path
    output_dir = os.path.dirname(output_df_path)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)
    output_tf_idf_path = args.output_tf_idf_path
    output_dir = os.path.dirname(output_tf_idf_path)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)

    # Подгружаем словарь в память
    token2id = load_dict(input_dict_path)
    # Находим инвертированный словарь
    id2token = {idx: token for token, idx in token2id.items()}
    # Считаем матрицу TF и вектор DF
    term_frequencies_sparse_matrix, documents_frequencies = get_df_sparse_tf_matrices_from_file(
        documents_path=input_documents_path, token2id=token2id)
    tf_idf_sparse_matrix = calculate_sparse_tf_idf_matrix(term_frequencies_sparse_matrix, documents_frequencies)
    # Записываем вектор DF (документные частоты терминов) в файл
    save_df_matrix(save_path=output_df_path, df=documents_frequencies, id2token=id2token)
    # Записываем матрицу TF-IDF в файл
    save_tf_idf_matrix(save_path=output_tf_idf_path, tf_idf_sparse_matrix=tf_idf_sparse_matrix, id2token=id2token)


if __name__ == '__main__':
    main()
