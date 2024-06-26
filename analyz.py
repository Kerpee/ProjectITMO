import json
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import lyricsgenius
import pymorphy3
from nltk.corpus import stopwords
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from googletrans import Translator
import matplotlib.pyplot as plt
from fuzzywuzzy import process, fuzz
from dotenv import load_dotenv
import shutil
from joblib import Memory

load_dotenv(os.path.join(os.getcwd(), 'api_token.env'))
api_token = os.getenv('api_token')
genius = lyricsgenius.Genius(api_token,
                             skip_non_songs=True,
                             excluded_terms=["(Remix)", "(Live)"])
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
morph = pymorphy3.MorphAnalyzer()
mem = Memory('cachedir', verbose=0)
mem_trans = Memory('trans_cache', verbose=0)


@mem_trans.cache()
def translate_text(text):  # Функция, необходимая для перевода слов для оценки эмоциональной окраски
    """
    Функция для перевода песен для анализа эмоциональной окраски песен
    @param  text:Полный текст всех песен исполнителя.
    @return Полный переведенный текст всех песен автора в одном файле
    """
    translator = Translator()
    translation = translator.translate(text, dest='en')
    return translation.text


def down(author):  # Скачивание автора, которого нет в базе
    """
    Функция для скачивания информации о песнях с Genius
    @param  author:Автор песен, которого необходимо проанализировать
    """
    artist = genius.search_artist(author)
    artist.save_lyrics(extension='json', overwrite=True, filename=f"Lyrics_{author}")
    direct = os.path.join(os.getcwd(), 'lyrics')
    filename = f"Lyrics_{author}.json"
    if os.path.exists(os.path.join(direct, filename)):
        os.remove(os.path.join(direct, filename))
    shutil.move(filename, direct)


# Проверка на совпадение автора, что позволяет уменьшить количество ошибок при вводе


def get_correct(obj, list_of_obj):
    """
    Функция для помощи пользователю в нахождении исполнителя, если он неправильно написал автора
    @param  obj:Объект, которого необходимо проанализировать
    @param list_of_obj: Список всех объектов, которые скачаны
    @return Возвращается правильное написание автора, если было превышен порог распознования, иначе возвращается вариант
    пользователя
       """

    obj_cool, score = process.extractOne(obj, list_of_obj, scorer=fuzz.token_sort_ratio)
    threshold = 42
    return obj_cool if score > threshold else obj


def inf(author):  # Создаём DataFrame файл, который содежрит информацию о песнях.
    """
       Функция для преобразования файла с json в DataFrame
       @param  author:Автор песен, которого необходимо проанализировать
       @return Переменная типа DataFrame с информацией о песнях
       """
    with open(os.path.join('lyrics', f"Lyrics_{author}.json")) as f:
        data_open = json.load(f)
        songs = []
        for row in data_open["songs"]:
            acc_dict = {}
            try:
                acc_dict['title'] = row['full_title']
            except KeyError:
                acc_dict['title'] = None
            try:
                acc_dict['lyrics_state'] = row['lyrics_state']
            except KeyError:
                acc_dict['lyrics_state'] = None
            try:
                acc_dict['release'] = row['release_date']
            except KeyError:
                acc_dict['release'] = None
            try:
                acc_dict['album'] = row['album']['name']
            except (KeyError, TypeError):
                acc_dict['album'] = None
            try:
                acc_dict['lyrics'] = row['lyrics']
            except KeyError:
                acc_dict['lyrics'] = None
            songs.append(acc_dict)
    df = pd.DataFrame(songs)
    df.dropna(inplace=True)
    df.head()
    songs_b = df.groupby(df.album).size().reset_index(name='counts')  # Сортируем песни по албомам исполнителя
    songs_b.sort_values(by="counts", ascending=False)
    songs_b.drop(songs_b[songs_b.counts < 1].index, inplace=True)
    songs_b.sort_values(by=["counts"], inplace=True, ascending=False)
    return df


def count_words(data, wordik):
    """
       Функция для подсчёта слов во всех песнях автора
       @param  data:Переменная типа DataFrame с информацией о песнях
       @param wordik:Слово, которое необходимо подсчитать
       @return Список, который отображает сколько раз используется нужное нам слово в каждой песне
       """
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
    """
       Функция для визуализации функции count_words
        @param  data:Переменная типа DataFrame с информацией о песнях
        @param list_of_word: Список, который отображает сколько раз используется нужное нам слово в каждой песне
        @param wordik:Слово, которое необходимо подсчитать
       """
    data[f"Количество {wordik}"] = list_of_word
    d_sort = data.sort_values(by=[f"Количество {wordik}"], ascending=False)
    data = d_sort.head(5)
    plt.figure(figsize=(12, 16))
    plt.bar(data['title'], data[f"Количество {wordik}"])
    plt.xlabel('Название')
    plt.ylabel('Количество слов')
    plt.show()


def max_words_album(album, data):  # Функция для подсчёта слов в альбоме
    """
           Функция для подсчёта слов во всех песнях автора
           @param album:Альбом, в котором необходимо подсчитать количество слов
           @param  data:Переменная типа DataFrame с информацией о песнях

           @return Список, который отображает сколько раз используется нужное нам слово в каждой песне в альбоме
           """
    list_of_albums = data['album']
    list_of_albums = list_of_albums.tolist()
    correct_album = get_correct(album, list_of_albums)
    album_songs = data[data['album'] == correct_album]['lyrics']
    vectorizer = CountVectorizer(
        stop_words=stopwords.words('russian'))  # Присуждаем каждому слову номер для облегчения подсчёта
    data_set = vectorizer.fit_transform(album_songs)
    sum_cnt = data_set.sum(axis=0)
    vocab = vectorizer.vocabulary_
    list_of_tuple = vocab.items()
    master_list = [(word, sum_cnt[0, index]) for word, index in list_of_tuple]
    master_list.sort(key=lambda x: x[1], reverse=True)
    return len(master_list), master_list


def max_words_song(data):  # Функция для подсчёта всех слов
    """
    Функция для подсчёта слов во всех песнях автора
    @param  data:Переменная типа DataFrame с информацией о песнях
    @return Список, который отображает сколько раз используется каждое слово у автора
    """
    vectorizer = CountVectorizer(stop_words=stopwords.words('russian'))
    data_set = vectorizer.fit_transform(data.lyrics)
    sum_cnt = data_set.sum(axis=0)
    vocab = vectorizer.vocabulary_
    list_of_tuple = vocab.items()
    master_list = [(word, sum_cnt[0, index]) for word, index in list_of_tuple]
    master_list.sort(key=lambda x: x[1], reverse=True)
    return master_list, master_list[:100]


def coeff(data):  # Высчитываем коэффиценты слов, то есть то насколько часто автор использует это слово во всех песнях
    """
    Функция для подсчёта коэффицентов слов, то есть то насколько часто автор использует это слово во всех песнях
    @param  data:Переменная типа DataFrame с информацией о песнях
    @return Список 20 самых используемых слов
    """
    tfid_vect = TfidfVectorizer(stop_words=stopwords.words('russian'), max_df=.4,
                                min_df=5)  # Создаём матрицу,оценивая важность слов
    data_set = tfid_vect.fit_transform(data.lyrics)
    sum_cnt1 = data_set.sum(axis=0)
    list_of_tuple1 = tfid_vect.vocabulary_.items()
    master_list1 = [(word, sum_cnt1[0, index]) for word, index in list_of_tuple1]
    master_list1.sort(key=lambda x: x[1], reverse=True)
    format_master_list1 = [(word, f"{value:.2f}") for word, value in master_list1]
    return format_master_list1[:20]


@mem.cache()
def wc(data):  # Оцениваем эмоциональную окраску текстов
    """
       Функция для эмоционлаьной окраски текстов автора
       @param  data:Переменная типа DataFrame с информацией о песнях
       @return Средние значения количества каждой эмоции и вывод, кооторый следует из этого
       """
    list_of_word, _ = max_words_song(data)
    analyzer = SentimentIntensityAnalyzer()
    scores = {'pos': 0, 'neg': 0, 'neu': 0, 'comp': 0}
    for word in range(len(list_of_word)):
        vs = analyzer.polarity_scores(translate_text(str(list_of_word[word][0])))
        print("{:-<30} {}".format(list_of_word[word][0], str(vs)))
        scores['pos'] += vs['pos']
        scores['neg'] += vs['neg']
        scores['neu'] += vs['neu']
        scores['comp'] += vs['compound']
    len_texts = len(data)
    avg_scores = {}
    for i, j in scores.items():
        avg_scores[i] = j / len_texts
    if avg_scores['comp'] >= 0.5:
        return 'Исполнитель более позитивный', avg_scores
    elif avg_scores['comp'] <= -0.5:
        return 'Исполнитель более негативный', avg_scores
    else:
        return 'Исполнитель более нейтрален', avg_scores


def cloud(data):  # Создаём кластеры слов автора
    """
       Функция для создания изображения с кластером слов автора
       @param  data:Переменная типа DataFrame с информацией о песнях
       """
    data = ' '.join(data['lyrics'])
    wordclouds = WordCloud(width=800, height=800, background_color='white',
                           stopwords=stopwords.words('russian')).generate(data)
    plt.figure(figsize=(10, 8))
    plt.imshow(wordclouds)
    plt.axis('off')
    plt.show()


def compare(data1, data2):
    """

    @param data1: Информация про первого автора
    @param data2: Информация про второго автора
    @return: Вызывает функцию для визуализации
    """
    author1 = None
    author2 = None
    for i in range(100):
        try:
            author1 = data1['title'][i].split('by')[-1]
            break
        except KeyError:
            continue
    for i in range(100):
        try:
            author2 = data2['title'][i].split('by')[-1]
            break
        except KeyError:
            continue
    unique_compare_2authors(author1, author2, data1, data2)


def unique_compare_2authors(author1, author2, data1, data2):
    """

    @param author1: Имя исполнителя
    @param author2: Имя второго исполнителя
    @param data1: Информация про 1 исполнителя
    @param data2: Информация про 2 исполнителя
    """
    unique_words1 = []
    unique_words2 = []

    for song in data1['lyrics']:
        unique_words1.append(' '.join(song.split()))

    for song in data2['lyrics']:
        unique_words2.append(' '.join(song.split()))
    cnt_of_w1 = len(set(''.join(unique_words1).split(' ')))
    cnt_of_w2 = len(set(''.join(unique_words2).split(' ')))
    df_unique_words = pd.DataFrame({
        'Исполнитель': [author1, author2],
        'Количество слов': [cnt_of_w1, cnt_of_w2]
    })

    df_unique_words.plot(x='Исполнитель', y='Количество слов', kind='bar', figsize=(12, 10))
    plt.title('Общее количество уникальных слов')
    plt.xlabel('Автор')
    plt.ylabel('Количество уникальных слов')
    plt.show()
