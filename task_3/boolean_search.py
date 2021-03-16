import codecs
import os
from argparse import ArgumentParser
from typing import List, Set

from task_3.utils import load_dict, load_inverted_index_from_file


# TODO: Возможно, добавить обработку NOT
def find_documents_in_index_by_word(word_id: int, inverted_index: List[List[int]], num_documents: int,
                                    get_present: bool = False) -> Set[int]:
    """
    :param word_id: Идентификатор слова, для которого находится список документов, в
    которых оно содержится/не содержится (содержание/не содержание зависит от значения
    параметра get_present)
    :param inverted_index: Инвертированный индекс документов: список из <размер словаря>
    списков, содержащий идентификаторы документов, в которых содержится соответствующее
    слово
    :param num_documents: Число документов коллекции. Используется для того, чтобы взять
    дополнение списка документов до всей коллекции документов для нахождения документов,
    в которых не содержится указанное слово с идентификатором word_id.
    :param get_present: True для взятия списка идентификаторов документов, в которых
    содержится слово с идентификатором word_id, иначе берётся дополнение этого списка до
    всей коллекции
    :return: Список уникальных номеров документов без повторений, в которых содержится/
    не содержится слово с идентификатором word_id
    """
    documents_list = inverted_index[word_id]
    if not get_present:
        all_docs_set = set(range(num_documents))
        documents_list = all_docs_set.difference(documents_list)
    return documents_list


def combine_docs_sets(docs_sets_list: List[Set[int]], operation: str) -> Set[int]:
    """
    Получает на вход список наборов идентификаторов документов и применяет к
    ним либо операцию объединения, либо операцию пересечения.
    :param docs_sets_list: Список наборов идентификаторов документов,
    к которым применяется операция типа operation. Набор идентификаторов
    содержит уникальные идентификаторы без повторений.
    :param operation: Тип операции: union/intersection, которая применяется
    к набору идентификаторов документов: объединение/пересечение.
    :return: Результат выполнения операций на списке наборов идентификаторов
    документов - один набор идентификаторов документов.
    """
    if operation == "union":
        resulting_set = set.union(docs_sets_list)
    elif operation == "intersection":
        resulting_set = set.intersection(docs_sets_list)
    else:
        raise ValueError(f"Invalid set operation: {operation}")
    return resulting_set


def parse_request(request_string: str, ):
    union_strings = request_string.split('|')
    intersection_strings = [s.split('^') for s in union_strings]
    return intersection_strings


def process_intersection_subrequest(intersection_tokens_strings, inverted_index, token2id, num_documents):
    """
    :param intersection_tokens_strings:
    :param inverted_index: Инвертированный индекс документов: список из <размер словаря>
    списков, содержащий идентификаторы документов, в которых содержится соответствующее
    слово
    :param token2id:
    :param num_documents: Число документов в коллекции.
    :return:
    """
    intersection_doc_ids_list = []
    for token in intersection_tokens_strings:
        if token.startswith('~'):
            present_flag = False
            token = token.strip('~')
        else:
            present_flag = True
        token_id = token2id[token]
        token_doc_ids = find_documents_in_index_by_word(word_id=token_id, inverted_index=inverted_index,
                                                        num_documents=num_documents, get_present=present_flag)
        intersection_doc_ids_list.append(token_doc_ids)
    doc_ids_intersection = combine_docs_sets(intersection_doc_ids_list, operation="intersection")

    return doc_ids_intersection


def main():
    parser = ArgumentParser()
    # TODO: Help
    parser.add_argument('--input_inv_index_path', default=r"inverted_index/inv_index.txt", type=str,
                        help="Путь к файлу инвертированного индекса")
    parser.add_argument('--input_dict_path', default=r"../task_2/tokenized_texts/dict.txt", type=str,
                        help="Путь к файлу словаря")
    # TODO
    parser.add_argument('--input_reviews_file', default=r"", type=str,
                        help="")
    parser.add_argument('--output_inv_index_path', default=r"inverted_index/inv_index.txt", type=str,
                        help="Выходной путь файла инвертированного индекса")

    args = parser.parse_args()
    input_inv_index_path = args.input_inv_index_path
    input_dict_path = args.input_dict_path

    token2id = load_dict(input_dict_path)
    inverted_index = load_inverted_index_from_file(input_inv_index_path)


if __name__ == '__main__':
    main()
