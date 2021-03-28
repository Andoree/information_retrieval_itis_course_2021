import codecs
import math
from typing import Dict

from scipy.sparse import csr_matrix


def load_tf_idf_matrix_from_file(tf_idf_file_path: str, token2id: Dict[str, int], sep_str: str = "~~~") -> csr_matrix:
    """
    :param tf_idf_file_path: Путь к файлу с TF-IDF значениями слов документов
    :param token2id: Словарь: маппинг из термина в идентификатор слова в словаре
    :param sep_str: Разделитель между термином и его значениями IDF и TF-IDF
    :return: Разреженная TF-IDF матрица документов коллекции
    """
    with codecs.open(tf_idf_file_path, 'r', encoding="utf-8") as input_file:
        vocab_size = len(token2id.keys())
        row_indices = []
        col_indices = []
        tf_idf_values = []
        num_documents = 0
        for doc_id, line in enumerate(input_file):
            num_documents += 1
            token_occurrences = line.strip().split()
            for t_occ in token_occurrences:
                token_attrs = t_occ.split(sep_str)
                token = token_attrs[0]
                token_id = token2id[token]
                token_tf_idf = float(token_attrs[2])
                row_indices.append(doc_id)
                col_indices.append(token_id)
                tf_idf_values.append(token_tf_idf)

    tf_idf_sparse_matrix = csr_matrix((tf_idf_values, (row_indices, col_indices)),
                                      shape=(num_documents, vocab_size))
    return tf_idf_sparse_matrix


def load_vocab_idfs(vocab_dfs_path: str, token2id: Dict[str, int], num_documents: int) -> Dict[int, float]:
    """
    Возвращает значения инвертированных документных частот (log IDF) терминов, используя
    документные частоты из файла документных частот
    :param vocab_dfs_path: Путь к файлу документных частот терминов
    :param token2id: Словарь: маппинг из термина в идентификатор слова в словаре
    :param num_documents: Общее число документов в коллекции
    :return: Словарь: маппинг из номера термина в словаре в его значение IDF
    """
    token_idfs = {}
    with codecs.open(vocab_dfs_path, 'r', encoding="utf-8") as inp_file:
        for line in inp_file:
            token_attrs = line.strip().split()
            token = token_attrs[0]
            token_id = token2id[token]
            token_df = int(token_attrs[1])
            token_idf = math.log2(num_documents / token_df)
            token_idfs[token_id] = token_idf
    return token_idfs
