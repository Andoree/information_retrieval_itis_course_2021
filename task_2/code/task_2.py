import codecs
import os
from argparse import ArgumentParser
from typing import List

from natasha import (
    Segmenter,
    MorphVocab,
    NewsMorphTagger,
    NewsEmbedding,
    Doc
)


def get_lemmatized_doc(raw_text: str, segmenter: Segmenter, morph_tagger: NewsMorphTagger,
                       morph_vocab: MorphVocab) -> List[str]:
    """
    :param raw_text: Строка, состоящая из тексте непредобработанного документа
    :param segmenter: токенизатор библиотеки Natasha
    :param morph_tagger: Морфологический парсер библиотеки Natasha. Морфологический
    разбор необходим для лемматизации: лемматизатор библиотеки Natasha использует
    его для при лемматизации
    :param morph_vocab: Лемматизатор библиотеки Natasha
    :return: Список лемм слов исходного текста с отброшенными знаками пунктуации
    """
    lemmatized_tokens = []
    natasha_doc = Doc(raw_text)
    # токенизация
    natasha_doc.segment(segmenter)
    # морфологический парсинг
    natasha_doc.tag_morph(morph_tagger)
    for token in natasha_doc.tokens:
        # лемматизация токенов
        token.lemmatize(morph_vocab)
        if token.pos != "PUNCT":
            lemmatized_tokens.append(token.lemma)
    return lemmatized_tokens


def main():
    parser = ArgumentParser()
    parser.add_argument('--input_data_dir', default=r"../../task_1/reviews/reviews", type=str,
                        help="Директория непредобработанных текстов, в данном случае - отзывов")
    parser.add_argument('--output_dir', default=r"../tokenized_texts", type=str,
                        help="Выходная директория, в которой будут содержаться словарь"
                             "и файл с лемматизированными текстами")
    parser.add_argument('--output_dict_fname', default=r"dict.txt", type=str,
                        help="имя файла словаря")
    parser.add_argument('--output_documents_fname', default=r"documents.txt", type=str,
                        help="Имя файла с лемматизированными документами")
    args = parser.parse_args()

    input_data_dir = args.input_data_dir

    output_dir = args.output_dir
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)
    output_dict_fname = args.output_dict_fname
    output_documents_fname = args.output_documents_fname
    output_dict_path = os.path.join(output_dir, output_dict_fname)
    output_documents_path = os.path.join(output_dir, output_documents_fname)

    segmenter = Segmenter()
    morph_vocab = MorphVocab()
    emb = NewsEmbedding()
    morph_tagger = NewsMorphTagger(emb)
    # список списков лемм всех документов
    lemmatized_tokens_lists = []
    # словарь лемм
    lemmas_dictionary = set()
    for document_fname in os.listdir(input_data_dir):
        document_path = os.path.join(input_data_dir, document_fname)
        with codecs.open(document_path, 'r', encoding="utf-8") as review_file:
            document_raw_text = review_file.read()
            # получаем список лемм документа
            lemmatized_tokens = get_lemmatized_doc(raw_text=document_raw_text, segmenter=segmenter,
                                                   morph_tagger=morph_tagger, morph_vocab=morph_vocab)
            # Добавляем список лемм в список списков лемм всех документов
            lemmatized_tokens_lists.append(lemmatized_tokens)
            # обновляем словарь лемм
            lemmas_dictionary.update(lemmatized_tokens)
    # запись словаря в файл
    with codecs.open(output_dict_path, 'w+', encoding="utf-8") as dict_file:
        for token in lemmas_dictionary:
            dict_file.write(f"{token}\n")
    # Запись лемматизированных документов в файл
    with codecs.open(output_documents_path, 'w+', encoding="utf-8") as documents_file:
        for doc_lemmas_list in lemmatized_tokens_lists:
            documents_file.write(f"{' '.join(doc_lemmas_list)}\n")


if __name__ == '__main__':
    main()
