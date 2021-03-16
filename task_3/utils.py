import codecs


# TODO: Документация + typing
def load_dict(dict_file_path: str):
    token2id = {}

    with codecs.open(dict_file_path, 'r', encoding="utf-8") as dict_file:
        for idx, line in enumerate(dict_file):
            token2id[line.strip()] = idx
    return token2id


def load_inverted_index_from_file(input_inv_index_path : str):
    inverted_index = []
    with codecs.open(input_inv_index_path, 'r', encoding="utf-8") as inv_index_file:
        for i, line in inv_index_file:
            doc_ids = set((int(x) for x in line.strip().split()))
            inverted_index.append(doc_ids)
    return inverted_index
