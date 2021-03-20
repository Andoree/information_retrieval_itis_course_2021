import codecs
import os
from argparse import ArgumentParser

from task_3.utils import load_dict


def main():
    parser = ArgumentParser()
    parser.add_argument('--input_documents_path', default=r"../task_2/tokenized_texts/documents.txt", type=str,
                        help="Путь к файлу, содержащему токенизированные документы")
    parser.add_argument('--input_dict_path', default=r"../task_2/tokenized_texts/dict.txt", type=str,
                        help="Путь к словарю")
    parser.add_argument('--output_inv_index_path', default=r"inverted_index/inv_index.txt", type=str,
                        help="Выходной путь файла инвертированного индекса")

    args = parser.parse_args()
    input_documents_path = args.input_documents_path
    input_dict_path = args.input_dict_path
    output_inv_index_path = args.output_inv_index_path
    output_dir = os.path.dirname(output_inv_index_path)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)

    token2id = load_dict(input_dict_path)
    vocab_size = len(token2id.keys())
    inverted_index = [[] for i in range(vocab_size)]
    with codecs.open(input_documents_path, 'r', encoding="utf-8") as documents_file:
        for doc_id, doc_line in enumerate(documents_file):
            doc_tokens = doc_line.strip().split()
            doc_unique_token_ids = set((token2id[token] for token in doc_tokens))
            for token_id in doc_unique_token_ids:
                inverted_index[token_id].append(doc_id)
    with codecs.open(output_inv_index_path, 'w+', ) as inv_index_file:
        for token_ids_list in inverted_index:
            inv_index_file.write(f"{' '.join((str(x) for x in token_ids_list))}\n")


if __name__ == '__main__':
    main()
