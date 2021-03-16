import codecs
from typing import Dict, List, Set


def load_dict(dict_file_path: str) -> Dict[str, int]:
    """
    Загружает словарь, в данном случае - словарь лемм, из файла
    :param dict_file_path: путь до файла словаря, каждая строка которого
    содержит 1 слово.
    :return: Словарь {слово : идентификатор слова в словаре}
    """
    token2id = {}
    with codecs.open(dict_file_path, 'r', encoding="utf-8") as dict_file:
        for idx, line in enumerate(dict_file):
            token2id[line.strip()] = idx
    return token2id


def load_inverted_index_from_file(input_inv_index_path: str) -> List[Set[int]]:
    """
    :param input_inv_index_path: путь до файла, содержащего инвертированный индекс.
    Каждая строка файла соответствует одному слову из словаря и содержит разделенные
    пробелами номера документов, в которых содержится соответствующее слово
    :return: Инвертированный индекс в виде списка списков. Каждый вложенный список
    содержит идентификаторы документов, в которых содержится одно некоторое слово.
    """
    inverted_index = []
    with codecs.open(input_inv_index_path, 'r', encoding="utf-8") as inv_index_file:
        for i, line in enumerate(inv_index_file):
            doc_ids = set((int(x) for x in line.strip().split()))
            inverted_index.append(doc_ids)
    return inverted_index
