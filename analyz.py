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


def get_author(author, list_of_authors):
    """
    Функция для помощи пользователю в нахождении исполнителя, если он неправильно написал автора
    @param  author:Автор песен, которого необходимо проанализировать
    @param list_of_authors: Список всех авторов, которые скачаны
    @return Возвращается правильное написание автора, если было превышен порог распознования, иначе возвращается вариант
    пользователя
       """
    author_cool, score = process.extractOne(author, list_of_authors, scorer=fuzz.token_sort_ratio)
    threshold = 42
    return author_cool if score > threshold else author


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
    plt.figure(figsize=(12, 6))
    plt.bar(data['title'], data[f"Количество {wordik}"])
    plt.xlabel('Слово')
    plt.ylabel('Количество')
    plt.title('Количество слов')
    plt.show()


def max_words_album(album, data):  # Функция для подсчёта слов в альбоме
    """
           Функция для подсчёта слов во всех песнях автора
           @param album:Альбом, в котором необходимо подсчитать количество слов
           @param  data:Переменная типа DataFrame с информацией о песнях

           @return Список, который отображает сколько раз используется нужное нам слово в каждой песне в альбоме
           """
    album = data.loc[data.album == album]
    vectorizer = CountVectorizer(
        stop_words=stopwords.words('russian'))  # Присуждаем каждому слову номер для облегчения подсчёта
    data_set = vectorizer.fit_transform(album.lyrics)
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
    '''

    @param data1: Информация про первого автора
    @param data2: Информация про второго автора
    @return: Вызывает функцию для визуализации
    '''
    len_data1 = len(data1)
    len_data2 = len(data2)
    sent_aut_1, score_aut_1 = wc(data1)
    sent_aut_2, score_aut_2 = wc(data2)

    for i in score_aut_1:
        score_aut_1[i] /= len_data1
    for i in score_aut_2:
        score_aut_2[i] /= len_data2
    author1 = data1['title'][0].split('by')[-1]
    author2 = data2['title'][0].split('by')[-1]

    plot_sent_2authors(author1, author2, score_aut_1, score_aut_2)


def plot_sent_2authors(author1, author2, score1, score2):
    '''
    Визуализациия сравнения авторов
    @param author1: Имя первого автора
    @param author2: Имя второго автора
    @param score1: Оценка первого автора
    @param score2: Оценка второго автора
    '''
    df = pd.DataFrame({
        f'{author1}': [score1['pos'], score1['neg'], score1['neu'], score1['comp']],
        f'{author2}': [score2['pos'], score2['neg'], score2['neu'], score2['comp']],
        'Emo': ["Pos", 'Neg', 'Neu', 'Comp']
    }).set_index('Emo')
    df.plot(figsize=(10, 8), kind='bar')
    plt.title('Сравнение эмоций')
    plt.xlabel('Эмоции')
    plt.ylabel('Ср.Знач')
    plt.show()
