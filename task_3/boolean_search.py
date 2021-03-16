import codecs
import os
from argparse import ArgumentParser
from typing import List

from task_3.utils import load_dict, load_inverted_index_from_file

# TODO: Возможно, добавить обработку NOT
def find_documents_in_index_by_word(word_id : int, inverted_index : List[List[int]], ):
    documents_list = inverted_index[word_id]
    return documents_list



def main():
    parser = ArgumentParser()
    # TODO: Help
    parser.add_argument('--input_inv_index_path', default=r"inverted_index/inv_index.txt", type=str,
                        help="")
    parser.add_argument('--input_dict_path', default=r"../task_2/tokenized_texts/dict.txt", type=str,
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
