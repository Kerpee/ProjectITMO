import json
import pandas
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import lyricsgenius
import pymorphy3
import stopwords
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from googletrans import Translator
import matplotlib.pyplot as plt
from fuzzywuzzy import process, fuzz

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
morph = pymorphy3.MorphAnalyzer()
stop = stopwords.get_stopwords('ru')


def translate_text(text):  # Функция, необходимая для перевода слов для оценки эмоциональной окраски
    translator = Translator()
    translation = translator.translate(text, dest='en')
    return translation.text


def down(author):  # Скачивание автора, которого нет в базе
    genius = lyricsgenius.Genius("Yj6AKvZjM2BDv-OMemcZ_cSCz2fubrAAB3PPV743bUpo1nzfAknSO0lKj-P_Jmrt",
                                 skip_non_songs=True,
                                 excluded_terms=["(Remix)", "(Live)"])
    artist = genius.search_artist(author)
    artist.save_lyrics()

# Проверка на совпадение автора, что позволяет уменьшить количество ошибок при вводе


def get_author(author, list_of_authors):
    author_cool, score = process.extractOne(author, list_of_authors, scorer=fuzz.token_sort_ratio)
    return author_cool if score > 42 else author


def inf(autohr):  # Создаём DataFrame файл, который содежрит информацию о песнях.
    with open(f"Lyrics_{autohr}.json") as f:
        datao = json.load(f)
        listi = []
        for row in datao["songs"]:
            acc_dict = {}
            try:
                acc_dict['title'] = row['full_title']
            except:
                acc_dict['title'] = None
            try:
                acc_dict['lyrics_state'] = row['lyrics_state']
            except:
                acc_dict['lyrics_state'] = None
            try:
                acc_dict['release'] = row['release_date']
            except:
                acc_dict['release'] = None
            try:
                acc_dict['album'] = row['album']['name']
            except:
                acc_dict['album'] = None
            try:
                acc_dict['lyrics'] = row['lyrics']
            except:
                acc_dict['lyrics'] = None
            listi.append(acc_dict)
    df = pandas.DataFrame(listi)
    df.dropna(inplace=True)
    df.head()
    songs_b = df.groupby(df.album).size().reset_index(name='counts')  # Сортируем песни по албомам исполнителя
    songs_b.sort_values(by="counts", ascending=False)
    songs_b.drop(songs_b[songs_b.counts < 1].index, inplace=True)
    songs_b.sort_values(by=["counts"], inplace=True, ascending=False)
    return df


def count_words(data, wordik):
    list_of_w = []
    w = wordik.lower()  # Уменьшаем регистр каждого слова для облегчения подсчета
    for index, row in data.iterrows():
        cnt = 0
        lyrics = row['lyrics'].lower().split()  # Разделяем каждое слово для его подсчёта
        for word in lyrics:
            if word == w:
                cnt += 1
        list_of_w.append(cnt)
    return list_of_w


def visual(data, list_of_word, wordik):  # Визуализация подсчёта слов
    data[f"Количество {wordik}"] = list_of_word
    d_sort = data.sort_values(by=[f"Количество {wordik}"], ascending=False)
    data = d_sort.head(5)
    plt.figure(figsize=(12, 6))
    plt.bar(data['title'], data[f"Количество {wordik}"])
    plt.xlabel('Слово')
    plt.ylabel('Количество')
    plt.title('Количество слов')
    plt.show()


def max_words_album(album, data):  # Функция для подсчёта слов в альбоме
    album = data.loc[data.album == album]
    vectorizer = CountVectorizer(stop_words=stop)  # Присуждаем каждому слову номер для облегчения подсчёта
    data_set = vectorizer.fit_transform(album.lyrics)
    sum_cnt = data_set.sum(axis=0)
    vocab = vectorizer.vocabulary_
    list_of_tuple = vocab.items()
    master_list = [(word, sum_cnt[0, index])for word, index in list_of_tuple]
    master_list.sort(key=lambda x: x[1], reverse=True)
    return master_list, len(master_list)


def max_words_song(data):  # Функция для подсчёта всех слов
    vectorizer = CountVectorizer(stop_words=stop)
    data_set = vectorizer.fit_transform(data.lyrics)
    sum_cnt = data_set.sum(axis=0)
    vocab = vectorizer.vocabulary_
    list_of_tuple = vocab.items()
    master_list = [(word, sum_cnt[0, index])for word, index in list_of_tuple]
    master_list.sort(key=lambda x: x[1], reverse=True)
    print(len(data.lyrics))
    return master_list


def coeff(df):  # Высчитываем коэффиценты слов, то есть то насколько часто автор использует это слово во всех песнях
    tfid_vect = TfidfVectorizer(stop_words=stop, max_df=.4, min_df=5)  # Создаём матрицу,оценивая важность слов
    data_set = tfid_vect.fit_transform(df.lyrics)
    sum_cnt1 = data_set.sum(axis=0)
    list_of_tuple1 = tfid_vect.vocabulary_.items()
    master_list1 = [(word, sum_cnt1[0, index])for word, index in list_of_tuple1]
    master_list1.sort(key=lambda x: x[1], reverse=True)
    format_master_list1 = [(word, f"{value:.2f}") for word, value in master_list1]
    return format_master_list1[:20]


def wc(data):  # Оцениваем эмоциональную окраску текстов
    list_of_word = max_words_song(data)
    analyzer = SentimentIntensityAnalyzer()
    for word in range(len(list_of_word)):
        vs = analyzer.polarity_scores(translate_text(str(list_of_word[word][0])))
        print("{:-<30} {}".format(list_of_word[word][0], str(vs)))


def cloud(data):  # Создаём кластеры слов автора
    data = ' '.join(data['lyrics'])
    wordclouds = WordCloud(width=800, height=800, background_color='white', stopwords=stop).generate(data)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordclouds, interpolation='bilinear')
    plt.axis('off')
    plt.show()
