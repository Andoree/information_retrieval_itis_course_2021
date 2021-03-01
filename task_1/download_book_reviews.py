import codecs
import os
import re
from argparse import ArgumentParser
from typing import List, Tuple

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


def get_review_texts_by_url(relative_reviews_urls: list, session: requests.sessions.Session, root_url: str) -> List[
    Tuple[str, str]]:
    """
    :param relative_reviews_urls: Список относительных URL относительно
    root_url.
    :param session: Сессия
    :param root_url: префикс любого URL отзыва на некоторую книгу
    :return: Список, состоящий из пар (абсолютный URL, текст книги)
    """
    texts = []
    for relative_url in tqdm(relative_reviews_urls):
        review_url = f"{root_url}/{relative_url}"
        response = session.get(review_url, headers={"User-Agent": "Mozilla/5.0"})
        response.encoding = "utf-8"
        response = response.text
        response = BeautifulSoup(response, "html.parser")
        text = response.find("div", {"class": "universal-blocks-content"}).text

        text = re.sub(f"[\t ]+", " ", text)

        texts.append((review_url, text))

    return texts


def main():
    parser = ArgumentParser()
    parser.add_argument('--root_url', default=r"https://bookmix.ru", type=str)
    parser.add_argument('--num_reviews', default=150, type=int)
    parser.add_argument('--save_dir', default=r"reviews/", type=str)
    parser.add_argument('--review_prefix', default=r"review", type=str)
    parser.add_argument('--index_fname', default=r"index.txt", type=str)

    args = parser.parse_args()
    session = requests.session()
    book_urls = []

    root_url = args.root_url
    num_reviews = args.num_reviews
    review_prefix = args.review_prefix
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    index_fname = args.index_fname
    print("Finding reviews urls.........")
    page_offset = 0
    while len(book_urls) < num_reviews:
        page_offset += 10
        review_search_url = f'https://bookmix.ru/reviews.phtml?option=all&begin={page_offset}&num_point=10&num_points=10'
        response = session.get(review_search_url, headers={'User-Agent': 'Mozilla/5.0'})
        response.encoding = 'utf-8'
        page = response.text
        soup = BeautifulSoup(page, 'html.parser')
        titles = soup.find_all('div', {'class': 'universal-blocks'})

        for title in titles:
            review_relative_url = title.find('h5').find('a').attrs['href']
            book_urls.append(review_relative_url)

    print(f"Successfully found {len(book_urls)} reviews")
    print("Loading reviews.........")
    texts = get_review_texts_by_url(book_urls, session, root_url)
    with codecs.open(os.path.join(save_dir, index_fname, ), 'w+', encoding="utf-8") as index_file:
        for i, t in enumerate(texts):
            with codecs.open(os.path.join(save_dir, f"{review_prefix}_{i}.txt", ), 'w+', encoding="utf-8") as text_file:
                text_file.write(f"{t[1].strip()}\n")
                index_file.write(f"{i}\t{t[0].strip()}\n")
        print(f"Successfully loaded {len(book_urls)} reviews")


if __name__ == '__main__':
    main()
