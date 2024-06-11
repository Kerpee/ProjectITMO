import tkinter as tk
from tkinter import ttk, messagebox
import os
from analyz import *

# Начальный класс, получающий начальную информацию об операции


class Analyzer:
    def __init__(self, author):
        self.author = author
        self.data = None
        self.down_data()
        
    def down_data(self):  # Функция, скачивающая текста автора, если таких нет, иначе собирает информацию
        author_files = [f for f in os.listdir() if f.startswith('Lyrics_') and f.endswith('.json')]
        authors_list = [f[7:-5] for f in author_files]
        self.author = get_author(self.author, authors_list)
        if not os.path.exists(f'Lyrics_{self.author}.json'):
            messagebox.showinfo('ОЙ','Не знали о таком исполнителе; Сейчас узнаем')
            down(self.author)
        self.data = inf(self.author)

# Класс, анализирующий тексты песней


class LyricAnalyzer(Analyzer):
    def count_russian_word(self, word, country):  # Специальная функция для подсчёта слов, учитывая падежи
        if country == "о":
            word_morph = morph.parse(word)[0]
            morphling = ["nomn", 'gent', 'ablt', 'accs', 'loct']
            mass_word = count_words(self.data, word)
            repl = []
            for i in range(len(morphling)):  # Предусмотрение случая, когда падежи дают одинаковые слова
                for j in range(len(morphling[i + 1:])):
                    if word_morph.inflect({morphling[i]}).word == word_morph.inflect({morphling[j]}).word:
                        repl.append(morphling[i])
            res = [x for x in morphling if not any([x.find(y) >= 0 for y in set(repl)])]
            for i in res:
                q = count_words(self.data, word_morph.inflect({i}).word)
                for j in range(len(mass_word)):
                    mass_word[j] = mass_word[j] + q[j]
            visual(self.data, mass_word, word)
        else:
            visual(self.data, count_words(self.data, word), word)

#  Вызов функций из analyz.py для необоходимых операций
    def count_words_in_album(self, album):
        return max_words_album(album, self.data)

    def calculate_importance_coefficient(self):
        return coeff(self.data)

    def count_all_words(self):
        return max_words_song(self.data)

    def analyze_word_emotion(self):
        return wc(self.data)

    def visualize_words(self):
        cloud(self.data)


class LyricsAnalyzerApp(tk.Tk):  # Класс, инициализирующая интерфейс программы
    def __init__(self):
        super().__init__()
        self.album_entry = None
        self.word_entry = None
        self.action_menu = None
        self.author_entry = None
        self.album_var = None
        self.word_var = None
        self.action_var = None
        self.author_var = None
        self.country_var = None
        self.title("Приложение для анализа")
        self.geometry('700x400')
        self.resizable(False, False)
        self.create_buttons()
        
    def create_buttons(self):  # Функция для создания интерактивных элементов
        frame = ttk.Frame(self, padding="10")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.country_var = tk.StringVar()
        self.author_var = tk.StringVar()
        self.action_var = tk.StringVar()
        self.word_var = tk.StringVar()
        self.album_var = tk.StringVar()

        ttk.Label(frame, text="Автор западный или отечественный?").grid(column=1, row=1, sticky=tk.W)
        ttk.Radiobutton(frame, text="Западный", variable=self.country_var, value="з").grid(column=2, row=1, sticky=tk.W)
        ttk.Radiobutton(frame, text="Отечественный", variable=self.country_var, value="о").grid(column=3, row=1,
                                                                                                sticky=tk.W)

        ttk.Label(frame, text="Введите автора для анализа:").grid(column=1, row=2, sticky=tk.W)
        self.author_entry = ttk.Entry(frame, width=20, textvariable=self.author_var)
        self.author_entry.grid(column=2, row=2, columnspan=2, sticky=(tk.W, tk.E))

        ttk.Label(frame, text="Что вы хотите сделать:").grid(column=1, row=3, sticky=tk.W)
        self.action_menu = ttk.Combobox(frame, textvariable=self.action_var)
        self.action_menu['values'] = ("Подсчёт опредленного слова",
                                      "Подсчёт всех слов альбоме",
                                      "Высчитать коэффицент важности",
                                      "Подсчёт всех слов исполнителя",
                                      "Высчёт эмоциональной окраски слова",
                                      'Показать визуализацию слов')
        self.action_menu.grid(column=2, row=3, columnspan=2, sticky=(tk.W, tk.E))

        ttk.Label(frame, text="Введите слово для подсчета:").grid(column=1, row=4,sticky=tk.W)
        self.word_entry = ttk.Entry(frame, width=20, textvariable=self.word_var)
        self.word_entry.grid(column=2, row=4, columnspan=2, sticky=(tk.W, tk.E))

        ttk.Label(frame, text="Введите название альбома:").grid(column=1, row=5, sticky=tk.W)
        self.album_entry = ttk.Entry(frame, width=20, textvariable=self.album_var)
        self.album_entry.grid(column=2, row=5, columnspan=2, sticky=(tk.W, tk.E))

        ttk.Button(frame, text="Начать анализ", command=self.start_analyzis).grid(column=2, row=6, columnspan=2)

    def start_analyzis(self):  # Функция, выводящая результаты на экран пользователя
        country = self.country_var.get()
        author = self.author_var.get()
        action = self.action_var.get()
        word = self.word_var.get()
        album = self.album_var.get()
        if not author:
            messagebox.showerror("Ошибка", "Введите имя автора.")
            return

        analyzer = LyricAnalyzer(author)

        if action == "Подсчёт опредленного слова":
            if not word:
                messagebox.showerror("Ошибка", "Введите слово для подсчета.")
                return
            analyzer.count_russian_word(word, country)

        elif action == "Подсчёт всех слов альбоме":
            if not album:
                messagebox.showerror("Ошибка", "Введите название альбома.")
                return
            result = analyzer.count_words_in_album(album)
            messagebox.showinfo("Результат", result)

        elif action == "Высчитать коэффицент важности":
            result = analyzer.calculate_importance_coefficient()
            messagebox.showinfo("Результат", result)

        elif action == "Подсчёт всех слов исполнителя":
            result = analyzer.count_all_words()
            messagebox.showinfo("Результат", result)

        elif action == "Высчёт эмоциональной окраски слова":
            result = analyzer.analyze_word_emotion()
            messagebox.showinfo("Результат", result)

        elif action == "Показать визуализацию слов":
            analyzer.visualize_words()


if __name__ == "__main__":
    app = LyricsAnalyzerApp()
    app.mainloop()
