import codecs
import os
from argparse import ArgumentParser

from task_3.utils import load_dict
from task_5.process_request import vectorize_request_tf_idf
from task_5.utils import load_tf_idf_matrix_from_file, load_vocab_idfs, load_doc_id_url_mapping_from_index
from natasha import Segmenter, NewsMorphTagger, MorphVocab, NewsEmbedding
from sklearn.metrics.pairwise import cosine_similarity

def main():
    parser = ArgumentParser()
    # parser.add_argument('--input_request_str', default=r"Классическая литература", type=str,
    #                     help="Строка поискового запроса")
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
    parser.add_argument('--input_documents_index', default=r"../task_1/reviews/index.txt", type=str,
                        help="Путь к индекс-файлу коллекции, содержащему маппинг номеров"
                             "документов в URL этих документов")

    args = parser.parse_args()
    input_dict_path = args.input_dict_path
    input_df_path = args.input_df_path
    input_tf_idf_path = args.input_tf_idf_path
    input_raw_documents_dir = args.input_raw_documents_dir
    input_documents_index = args.input_documents_index
    # output_log_path = args.output_log_path
    # output_dir = os.path.dirname(output_log_path)
    # if not os.path.exists(output_dir) and output_dir != '':
    #     os.makedirs(output_dir)

    # Подгружаем словарь в память
    token2id = load_dict(input_dict_path)
    # Подгружаем предпосчитанную матрицу TF-IDF из файла
    tf_idf_matrix = load_tf_idf_matrix_from_file(tf_idf_file_path=input_tf_idf_path, token2id=token2id, )
    num_documents = tf_idf_matrix.shape[0]
    # Подгружаем инвертированные документные частоты терминов (IDF)
    token_idfs = load_vocab_idfs(vocab_dfs_path=input_df_path, token2id=token2id, num_documents=num_documents)
    doc_id2url = load_doc_id_url_mapping_from_index(input_documents_index)

    segmenter = Segmenter()
    morph_vocab = MorphVocab()
    emb = NewsEmbedding()
    morph_tagger = NewsMorphTagger(emb)

    while True:
        # Принимаем текст запроса пользователя
        input_request_str = input("Введите поисковый запрос:\n")
        if input_request_str == '-1':
            break
        # Векторизуем запрос в разреженный TF-IDF вектор
        request_vector = vectorize_request_tf_idf(request_raw_text=input_request_str, segmenter=segmenter,
                                                  morph_tagger=morph_tagger, morph_vocab=morph_vocab, token2id=token2id,
                                                  token_idfs=token_idfs)
        # Идентификатор документа, наиболее похожего на запрос векторно. Мера похожести - косинусная близость векторов
        response_document_id = cosine_similarity(tf_idf_matrix, request_vector).argmax()
        # Путь до файла исходного непредобработанного документа
        response_document_path = os.path.join(input_raw_documents_dir, f"review_{response_document_id}.txt")
        # Находим URL документа в индекса
        response_document_url = doc_id2url[response_document_id]
        with codecs.open(response_document_path, 'r', encoding="utf-8") as raw_text_file:
            document_text = raw_text_file.read().strip()
            print(f"Строка запроса: {input_request_str}")
            print(f"Номер документа - ответа на запрос: {response_document_id}")
            print(f"URL документа - ответа на запрос: {response_document_url}")
            print(f"Текст документа:\n---\n{document_text}\n---\n")


if __name__ == '__main__':
    main()