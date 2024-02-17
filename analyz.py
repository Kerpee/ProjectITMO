import json
import pandas
import pandas as pd
import altair as alt
import altair_viewer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import lyricsgenius
import os.path
import pymorphy3
import tkinter
import stopwords

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
morph=pymorphy3.MorphAnalyzer()
stop=stopwords.get_stopwords('ru')
stopen=stopwords.get_stopwords('en')

def down(author):
    genius = lyricsgenius.Genius("Yj6AKvZjM2BDv-OMemcZ_cSCz2fubrAAB3PPV743bUpo1nzfAknSO0lKj-P_Jmrt", skip_non_songs=True, excluded_terms=["(Remix)", "(Live)"])
    artist = genius.search_artist(author)
    artist.save_lyrics()


def inf(autohr):
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
    release = df['release'].str.extract(r'(\d{4})')
    df.release = release
    songs_b_y = df.groupby(df.album).size().reset_index(name='counts')
    songs_b_y.sort_values(by="counts", ascending=False)
    songs_b_y.drop(songs_b_y[songs_b_y.counts < 1].index, inplace=True)
    songs_b_y.sort_values(by=["counts"], inplace=True, ascending=False)
    return df


def count_words(data, wordik):
    list_of_w = []
    w = wordik.lower()
    for index, row in data.iterrows():
        cnt = 0
        lyrics = row['lyrics'].lower().split()
        for word in lyrics:
            if word == w:
                cnt += 1
        list_of_w.append(cnt)
    return list_of_w


def visual(data, list_of_word, wordik):
    data[f"Количество {wordik}"] = list_of_word
    data.sort_values(by=[f"Количество {wordik}"], ascending=False)
    chart = alt.Chart(data[:15]).mark_bar().encode(
        alt.X("title", sort=alt.EncodingSortField(field=f'Количество {wordik}', op='sum', order='descending')),
        alt.Y(f'Количество {wordik}')
    )
    altair_viewer.show(chart)


def max_words_album(album, data):
    album = data.loc[data.album == album]
    vectorizer = CountVectorizer(stop_words=stop)
    data_set = vectorizer.fit_transform(album.lyrics)
    # Album_df = pd.DataFrame(X.toarray(), index=album.title, columns=vectorizer.get_feature_names_out())
    sum_cnt = data_set.sum(axis=0)
    vocab = vectorizer.vocabulary_
    list_of_tuple = vocab.items()
    master_list = [(word, sum_cnt[0, index])for word, index in list_of_tuple]
    master_list.sort(key=lambda x: x[1], reverse=True)
    return master_list,len(master_list)


def max_words_song(data):
    vectorizer = CountVectorizer(stop_words=stop)
    data_set = vectorizer.fit_transform(data.lyrics)
    sum_cnt = data_set.sum(axis=0)
    vocab = vectorizer.vocabulary_
    list_of_tuple = vocab.items()
    master_list = [(word, sum_cnt[0, index])for word, index in list_of_tuple]
    master_list.sort(key=lambda x: x[1], reverse=True)
    print(len(data.lyrics))
    return len(master_list),int(len(master_list)/len(data.lyrics)),master_list


def coeff(df):
    tfid_vect = TfidfVectorizer(stop_words=stop, max_df=.2, min_df=4)
    data_set = tfid_vect.fit_transform(df.lyrics)
    sum_cnt1 = data_set.sum(axis=0)
    list_of_tuple1 = tfid_vect.vocabulary_.items()
    master_list1 = [(word, sum_cnt1[0, index])for word, index in list_of_tuple1]
    master_list1.sort(key=lambda x: x[1], reverse=True)
    return master_list1[:20]


def main():
    country = input("Автор западный(з) или отечественный(о)?")
    autohr = input("Введите автора для анализа:")
    if os.path.exists(f"Lyrics_{autohr}.json"):
        action = input("Что вы хотите сделать:\nПодсчитать количество использования определенного слова в альбоме автора(1)\nПодсчитать количество слов в альбоме(2)\nПодсчитать коэффицент важности слов для автора(3)\nПодсчитать общее количество слов в текстах автора(4)\n")

        if action == "1" and country == "о":
            data = inf(autohr)

            wordik = input("Какое слово вы хотите подсчитать:")
            word = morph.parse(wordik)[0]
            mass_word = count_words(data, wordik)
            morphling = ["nomn", 'gent', 'ablt', 'accs', 'loct']
            povtor = []
            for i in range(len(morphling)):
                for j in range(len(morphling[i+1:])):
                    if word.inflect({morphling[i]}).word == word.inflect({morphling[j]}).word:
                        povtor.append(morphling[i])
            print(povtor)
            res = [x for x in morphling if not any([x.find(y) >= 0 for y in set(povtor)])]
            print(res)
            for i in res:
                q = count_words(data, word.inflect({i}).word)
                for j in range(len(mass_word)):
                    mass_word[j] = mass_word[j]+q[j]
            visual(data, mass_word, wordik)
        elif action == "1" and country == "з":
            data = inf(autohr)
            word = input("Какое слово вы хотите подсчитать? ")
            visual(data, count_words(data, word), word)
        elif action == "2":

            data = inf(autohr)
            album = input('В каком альбоме? ')
            print(max_words_album(album, data))

        elif action == "3":

            data = inf(autohr)
            print(coeff(data))
        elif action == "4":
            data = inf(autohr)
            print(max_words_song(data))
    else:
        print("Не знали о таком исполнителе, сейчас узнаем...")
        down(autohr)
        main()


if __name__ == "__main__":
    main()
