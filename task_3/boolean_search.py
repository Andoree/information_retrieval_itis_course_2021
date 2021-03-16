from argparse import ArgumentParser
from typing import List, Set, Dict

from task_3.utils import load_dict, load_inverted_index_from_file


def find_documents_in_index_by_word(word_id: int, inverted_index: List[Set[int]], num_documents: int,
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
        resulting_set = set.union(*docs_sets_list)
    elif operation == "intersection":
        resulting_set = set.intersection(*docs_sets_list)
    else:
        raise ValueError(f"Invalid set operation: {operation}")
    return resulting_set


def get_request_intersection_units(request_string: str, ) -> List[List[str]]:
    """
    Выполняет первичную предобработку поискового запроса, разбивая его на подзапросы
    на совместное (не)появление некоторых слов в документах
    :param request_string: Строка запроса
    :return: Текстовые подзапросы, в которых содержатся только операции пересечения и
    дополнения (из теории множеств) и не содержатся операции объединения, поскольку
    исходный запрос представляет собой объединение подзапросов, возвращаемых этим методом.
    """
    union_strings = request_string.split('|')
    intersection_strings = [s.split('^') for s in union_strings]
    return intersection_strings


def process_intersection_subrequest(intersection_tokens_strings: List[str], inverted_index: List[Set[int]],
                                    token2id: Dict[str, int], num_documents: int) -> Set[int]:
    """
    Получает список слов и возвращает номера документов, в которых эти слова (не)встречаются
    совместно. Для случая непоявления слова в документе перед ним ставится символ '~'
    :param intersection_tokens_strings: Список строк - слов, от которых требуется одновременное
    (не)присутствие в документах
    :param inverted_index: Инвертированный индекс документов: список из <размер словаря>
    списков, содержащий идентификаторы документов, в которых содержится соответствующее
    слово
    :param token2id: Словарь {слово : номер его позиции в словаре}
    :param num_documents: Число документов в коллекции.
    :return: Набор уникальных идентификаторов документов, удовлетворяющих строке запроса на
    одновременное (не)присутствие в документе.
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


def get_doc_ids_by_request(request_string: str, inverted_index: List[Set[int]], token2id: Dict[str, int],
                           num_documents: int) -> Set[int]:
    """
    Получает на вход строку запроса, соответствующую введенному мной языку запросов,
    возвращает список номеров документов, удовлетворяющих этому запросу.
    :param request_string: Строка запроса
    :param inverted_index: Инвертированный индекс документов
    :param token2id: Словарь инвертированного индекса
    :param num_documents: Общее число документов в коллекции
    :return: Список уникальных номеров документов, удовлетворяющих полученному запросу
    """
    intersection_strings_list = get_request_intersection_units(request_string)
    union_units = []
    for intersection_strs in intersection_strings_list:
        doc_ids_intersection = process_intersection_subrequest(intersection_strs, inverted_index=inverted_index,
                                                               token2id=token2id, num_documents=num_documents)
        union_units.append(doc_ids_intersection)
    united_doc_ids = combine_docs_sets(union_units, operation="union")
    return united_doc_ids


def main():
    parser = ArgumentParser()
    parser.add_argument('--input_inv_index_path', default=r"inverted_index/inv_index.txt", type=str,
                        help="Путь к файлу инвертированного индекса")
    parser.add_argument('--input_dict_path', default=r"../task_2/tokenized_texts/dict.txt", type=str,
                        help="--num_documents")
    parser.add_argument('--num_documents', default=154, type=int, help="Число документов в коллекции")
    parser.add_argument('--request_string', default="иванов|ответ^бог", type=str,
                        help="Строка поискового запроса. Строка состоит из конъюнктов, разделенных символом '|'."
                             "Конъюнкт - набор лемм, которые должны (не должны) встретиться в документе совместно."
                             "Такие леммы разделены символом '^'. Если необходимо, чтобы леммы не было в документе,"
                             "перед ним ставится символ '~' без пробелов. Итого, примерный запрос выглядит так:"
                             "<лемма_1>^~<лемма_2>^<лемма_3>|<лемма_4>^<лемма_5>")


    args = parser.parse_args()
    input_inv_index_path = args.input_inv_index_path
    input_dict_path = args.input_dict_path
    request_string = args.request_string
    num_documents = args.num_documents

    # Подгружаем словарь в память
    token2id = load_dict(input_dict_path)
    # Подгружаем инвертированный индекс документов в память
    inverted_index = load_inverted_index_from_file(input_inv_index_path)
    # Выполняем поисковый запрос методом булева поиска
    request_result = get_doc_ids_by_request(request_string, inverted_index, token2id, num_documents)
    print(request_result)


if __name__ == '__main__':
    main()
