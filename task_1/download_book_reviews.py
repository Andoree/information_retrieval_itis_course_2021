import codecs
import os

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


def get_texts(hrefs, session, root):
    texts = []
    for href in tqdm(hrefs):
        href = f'{root}/{href}'
        req = session.get(href, headers={'User-Agent': 'Mozilla/5.0'})
        req.encoding = 'utf-8'
        req = req.text
        req = BeautifulSoup(req, 'html.parser')
        text = req.find('div', {'class': 'universal-blocks-content'}).text.lower().replace('\n', ' ').replace('\t', ' ')
        texts.append((href, text))

    return texts

def main():
    session = requests.session()
    book_urls = []
    begin = 0
    root_url = "https://bookmix.ru"
    output_texts_dir = "reviews"
    if not os.path.exists(output_texts_dir):
        os.makedirs(output_texts_dir)

    while len(book_urls) < 150:
        begin += 10
        url = f'https://bookmix.ru/reviews.phtml?option=all&begin={begin}&num_point=10&num_points=10'
        req = session.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        req.encoding = 'utf-8'
        page = req.text
        soup = BeautifulSoup(page, 'html.parser')
        titles = soup.find_all('div', {'class': 'universal-blocks'})

        for title in titles:
            href = title.find('h5').find('a').attrs['href']
            book_urls.append(href)
    texts = get_texts(book_urls, session, root_url)
    with codecs.open(os.path.join(output_texts_dir, "index.txt", ), 'w+', encoding="utf-8") as index_file:
        for i, t in enumerate(texts):
            with codecs.open(os.path.join(output_texts_dir, f"text_{i}.txt", ), 'w+', encoding="utf-8") as text_file:
                text_file.write(f"{t[1].strip()}\n")
                index_file.write(f"{i}\t{t[0].strip()}\n")




if __name__ == '__main__':
    main()
