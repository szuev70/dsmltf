#!/usr/bin/env python
# coding: utf-8

from math import sqrt, pi, exp, erf, log, gamma, floor
from copy import copy
from collections import Counter, defaultdict
import random
from matplotlib import pyplot as plt
from functools import partial
import re
from bs4 import BeautifulSoup as BS
import requests


def lastc(s):
    """
    Эта функция возвращает последний символ строки s

    Parameters
    ----------
        s (str): Строка

    Returns
    -------
        str: Последний символ строки s
    """
    return s[-1]


def app_to_you(fun):
    """
    Эта функция применяет функцию fun к строке 'you'

    Parameters
    ----------
        fun (function): Функция

    Returns
    -------
        any: Результат применения функции fun к строке 'you'
    """
    return fun('you')


def flastc(s='f'):
    """
    Эта функция возвращает функцию, которая возвращает последний символ строки s

    Parameters
    ----------
        s (str): Строка

    Returns
    -------
        str: Последний символ строки s
    """
    return lastc(s)


def lazy_range(n):
    """
    Ленивый range

    Parameters
    ----------
        n (int): Число

    Returns
    -------
        int: Число
    """
    i = 0
    while i < n:
        yield i
        i += 1


def natural_numbers():
    """
    Бесконечный генератор натуральных чисел

    Returns
    -------
        int: Число
    """
    n = 1
    while True:
        yield n
        n += 1


class Set:
    """
    Класс множества
    """

    def __init__(self, values=None):
        """
        Конструктор класса множества

        Parameters
        ----------
            values (list): Список значений
        """
        self.dict = {}
        if values is not None:
            for value in values:
                self.add(value)

    def __repr__(self):
        """
        Переопределение метода __repr__

        Returns
        -------
            str: Строковое представление множества
        """
        return "Set: " + str(self.dict.keys())

    def add(self, value):
        """
        Добавление элемента в множество

        Parameters
        ----------
            value (any): Элемент
        """
        self.dict[value] = True

    def contains(self, value):
        """
        Проверка наличия элемента в множестве

        Parameters
        ----------
            value (any): Элемент

        Returns
        -------
            bool: True, если элемент есть в множестве, иначе False
        """
        return value in self.dict

    def remove(self, value):
        """
        Удаление элемента из множества

        Parameters
        ----------
            value (any): Элемент
        """
        del self.dict[value]


def double(x):
    """
    Удвоение числа

    Parameters
    ----------
        x (int): Число

    Returns
    -------
        int: Удвоенное число
    """
    return 2*x


def multiply(x, y):
    """
    Умножение чисел

    Parameters
    ----------
        x (int): Число
        y (int): Число

    Returns
    -------
        int: Произведение чисел
    """
    return x*y


def is_even(x):
    """
    Проверка на четность

    Parameters
    ----------
        x (int): Число
        
    Returns
    -------
        bool: True, если число четное, иначе False
    """
    return x % 2 == 0


def miracle(*args, **kwargs):
    print("безымянные аргументы:", args)
    print("аргументы по ключу:", kwargs)


def vector_add(v, w):
    """
    Сложение векторов

    Parameters
    ----------
        v (list): Вектор
        w (list): Вектор

    Returns
    -------
        list: Сумма векторов
    """
    return [v_i + w_i for v_i, w_i in zip(v, w)]


def vector_sum(vectors):
    """
    Сумма векторов из списка векторов

    Parameters
    ----------
        vectors (list): Список векторов

    Returns
    -------
        list: Сумма векторов
    """
    return reduce(vector_add, vectors)


def scalar_multiply(c, v):
    """
    Умножение вектора на число

    Parameters
    ----------
        c (scalar): Число
        v (list): Вектор

    Returns
    -------
        list: Произведение вектора на число
    """
    return list(map(lambda x: c*x, v))


def dot(v, w):
    """
    Скалярное произведение векторов

    Parameters
    ----------
        v (list): Вектор
        w (list): Вектор

    Returns
    -------
        float: Скалярное произведение векторов
    """
    return sum(v_i * w_i for v_i, w_i in zip(v, w))


def vector_mean(vectors):
    """
    Покомпонентное среднее

    Parameters
    ----------
        vectors (list): Список векторов

    Returns
    -------
        list: Покомпонентное среднее
    """
    return scalar_multiply(1/len(vectors), vector_sum(vectors))


def sum_of_squares(v):
    """
    Квадрат модуля вектора

    Parameters
    ----------
        v (list): Вектор

    Returns
    -------
        float: Квадрат модуля вектора
    """
    return dot(v, v)


def magnitude(v):
    """
    Модуль вектора

    Parameters
    ----------
        v (list): Вектор

    Returns
    -------
        float: Модуль вектора
    """
    return sqrt(sum_of_squares(v))


def squared_distance(v, w):
    """
    Квадрат расстояния между векторами

    Parameters
    ----------
        v (list): Вектор
        w (list): Вектор

    Returns
    -------
        float: Квадрат расстояния между векторами
    """
    return sum_of_squares(vector_add(v,  scalar_multiply(-1, w)))


def distance(v, w):
    """
    Расстояние между векторами

    Parameters
    ----------
        v (list): Вектор
        w (list): Вектор

    Returns
    -------
        float: Расстояние между векторами
    """
    return sqrt(squared_distance(v, w))


def shape(A):
    """
    Размерность матрицы

    Parameters
    ----------
        A (list of list): Матрица

    Returns
    -------
        tuple: Размерность матрицы
    """
    n = len(A)
    m = len(A[0]) if A else 0
    return n, m


def get_column(A, i):
    """
    Получение столбца матрицы

    Parameters
    ----------
        A (list of list): Матрица
        i (int): Номер столбца

    Returns
    -------
        list: Столбец матрицы
    """
    return [a[i] for a in A]


def make_matrix(num_rows, num_cols, entry_fn):
    """
    Формирование матрицы как функции от позиции элемента

    Parameters
    ----------
        num_rows (int): Количество строк
        num_cols (int): Количество столбцов
        entry_fn (function): Функция от позиции элемента

    Returns
    -------
        list of list: Матрица
    """
    return [[entry_fn(i, j) for j in range(num_cols)]
            for i in range(num_rows)]


def mean(x):
    """
    Среднее значение

    Parameters
    ----------
        x (list): Список чисел

    Returns
    -------
        float: Среднее значение
    """
    return sum(x)/len(x)


def de_mean(x):
    """
    Вектор отклонений от среднего

    Parameters
    ----------
        x (list): Список чисел

    Returns
    -------
        list: Вектор отклонений от среднего
    """
    x_bar = mean(x)
    return [x_i - x_bar for x_i in x]


def median(x):
    """
    Медиана распределения. При четной выборке медианное значение выбрасывается!

    Parameters
    ----------
        x (list): Список чисел

    Returns
    -------
        float: Медиана
    """
    def closest(y, av):
        return min(y, key=lambda t: abs(t-av))
    avx = mean(x)
    cmx = closest(x, avx)
    if len(x) % 2:
        return cmx
    else:
        x.remove(cmx)
        return (cmx+closest(x, avx))/2


def quantile(x, p):
    """
    Квантиль распределения
    """
    p_i = int(p * len(x))
    return sorted(x)[p_i]


def data_range(x):
    """
    Размах распределения

    Parameters
    ----------
        x (list): Список чисел

    Returns
    -------
        float: Размах распределения
    """
    return max(x) - min(x)


def de_mean(x):
    """
    Отклонения от среднего

    Parameters
    ----------
        x (list): Список чисел

    Returns
    -------
        list: Отклонения от среднего
    """
    return [x_i - mean(x) for x_i in x]


def variance(x):
    """
    Дисперсия

    Parameters
    ----------
        x (list): Список чисел

    Returns
    -------
        float: Дисперсия
    """
    return mean(list(map(lambda t: t**2, x))) - mean(x)**2


def standard_deviation(x):
    """
    Стандартное отклонение

    Parameters
    ----------
        x (list): Список чисел

    Returns
    -------
        float: Стандартное отклонение
    """
    return sqrt(variance(x))


def interquantile_range(x):
    """
    Интерквантильный размах

    Parameters
    ----------
        x (list): Список чисел

    Returns
    -------
        float: Интерквантильный размах
    """
    return quantile(x, 0.75) - quantile(x, 0.25)


def covariance(x, y):
    """
    Ковариация

    Parameters
    ----------
        x (list): Список чисел
        y (list): Список чисел

    Returns
    -------
        float: Ковариация
    """
    return dot(de_mean(x), de_mean(y)) / len(x)


def correlation(x, y):
    """
    Корреляция

    Parameters
    ----------
        x (list): Список чисел
        y (list): Список чисел

    Returns
    -------
        float: Корреляция
    """
    if standard_deviation(x) > 0 and standard_deviation(y) > 0:
        return covariance(x, y)/standard_deviation(x)/standard_deviation(y)
    else:
        return 0


def rho_even(x, a, b):
    """
    Равномерное распределение на [a,b]

    Parameters
    ----------
        x (float): Значение
        a (float): Нижняя граница
        b (float): Верхняя граница

    Returns
    -------
        float: Значение плотности вероятности
    """
    return 1 / (b-a) if x >= a and x <= b else 0


def rho_norm(x, mu=0, s=1):
    """
    Нормальное распределение – плотность вероятности

    Parameters
    ----------
        x (float): Значение
        mu (float): Математическое ожидание
        s (float): Стандартное отклонение

    Returns
    -------
        float: Значение плотности вероятности
    """
    return 1/sqrt(2*pi)/s*exp(-(x-mu)**2/2/s**2)


def f_norm(x, mu=0, s=1):
    """
    Нормальное распределение – функция распределения

    Parameters
    ----------
        x (float): Значение
        mu (float): Математическое ожидание
        s (float): Стандартное отклонение

    Returns
    -------
        float: Значение функции распределения
    """
    return (1+erf((x-mu)/sqrt(2)/s))/2


def inv_f_norm(p, mu, s, t=0.001):
    """
    Обратная функция нормального распределения в полосе значений -100…100

    Parameters
    ----------
        p (float): Значение функции распределения
        mu (float): Математическое ожидание
        s (float): Стандартное отклонение
        t (float): Точность

    Returns
    -------
        float: Значение
    """
    # сначала перейдем к стандартному нормальному распределению
    if mu != 0 or s != 1:
        return mu + s * inv_f_norm(p, 0, 1, t)
    # ищем в полосе значений -100…100
    low_x, low_p = -100.0, 0
    hi_x, hi_p = 100.0, 1
    while hi_x - low_x > t:
        mid_x = (low_x + hi_x)/2
        mid_p = f_norm(mid_x)
        if mid_p < p:
            low_x, low_p = mid_x, mid_p
        elif mid_p > p:
            hi_x, hi_p = mid_x, mid_p
        else:
            break
    return mid_x


def p_value(x, mu=0, s=1):
    """
    p-значение

    Parameters
    ----------
        x (float): Значение
        mu (float): Математическое ожидание
        s (float): Стандартное отклонение

    Returns
    -------
        float: p-значение
    """
    if x >= mu:
        return 2*(1-f_norm(x, mu, s))
    else:
        return 2*f_norm(x, mu, s)


def two_side_p_value(x, mu, s):
    """
    Двустороннее p-значение

    Parameters
    ----------
        x (float): Значение
        mu (float): Математическое ожидание
        s (float): Стандартное отклонение

    Returns
    -------
        tuple: Двустороннее p-значение
    """
    return f_norm(x, mu, s), 1-f_norm(x, mu, s)


def approx_exp(x, t):
    """
    Аппроксимация экспоненты

    Parameters
    ----------
        x (list): Список чисел, только положительные больше 0
        t (list): Список чисел range(1, len(x)+1)

    Returns
    -------
        tuple: Параметры A и b
    """
    n = len(x)
    y = list(map(log, x))
    sum_t, sum_y = sum(t), sum(y)
    sum_t2 = sum(ti**2 for ti in t)
    sum_yt = sum(ti*yi for ti, yi in zip(t, y))
    a = (sum_yt*sum_t - sum_y*sum_t2) / (sum_t**2 - sum_t2*n)
    b = (sum_y*sum_t - sum_yt*n) / (sum_t**2 - sum_t2*n)
    return exp(a), b


def gauss_slae(A, b):
    """
    Метод Гаусса решения СЛАУ

    Parameters
    ----------
        A (list of list): Матрица коэффициентов
        b (list): Свободные члены

    Returns
    -------
        list: Решение
    """

    def ni(l, i):  
        """
        Нормирование списка l на единицу в позиции i
        
        Parameters
        ----------
            l (list): Список
            i (int): Позиция

        Returns
        -------
            list: Нормированный список
        """
        return [lj/l[i] for lj in l]

    def ch_stack(L, i):
        """
        Перемещение строки вниз матрицы (на ее место ставится следующая строка)

        Parameters
        ----------
            L (list of list): Матрица
            i (int): Номер строки

        Returns
        -------
            list of list: Матрица
        """
        L.append(L[i])
        L.pop(i)
        return L

    def corr_row(Gm, Gn, n):
        """
        Корректировка списка Gm списком Gn, нормированным на единицу в позиции n

        Parameters
        ----------
            Gm (list): Список
            Gn (list): Список
            n (int): Позиция

        Returns
        -------
            list: Список
        """
        gml = ni(Gm, n)
        return [gmlk - gnk for gmlk, gnk in zip(gml, Gn)]

    x = []      # инициируем список, который потом станет решением
    n = len(b)  # вычисляем порядок системы
    # строим расширенную матрицу системы
    G = [ai+[bi] for ai, bi in zip(A, b)]
    while n > 1:  # в этом цикле нормируем строки на их диагональный элемент
        n -= 1
        if not G[n][n]:
            ch_stack(G, n)
        cGn = copy(G[n])
        G[n] = ni(cGn, n)
        m = n
        while m > 0:  # в этом цикле корректируем все строки, выше той, что 
                      # только что отнормирована
            m -= 1
            if G[m][n]:
                cGm = copy(G[m])
                G[m] = corr_row(cGm, G[n], n)
    # прямой проход закончен
    # присваиваем значение первому неизвестному
    x.append(G[0][-1]/G[0][0])
    for gi in G[1:]:  # последовательно вычисляем все остальные неизвестные
        x.append((gi[-1]-dot(x, gi))/gi[len(x)])
    return x


def approx_poly(x, t, r):
    """
    Аппроксимация полиномом

    Parameters
    ----------
        x (list): Список чисел
        t (list): Список чисел, range(1, len(x)+1)
        r (int): Степень полинома

    Returns
    -------
        list: Параметры полинома
    """
    M = [[] for _ in range(r+1)]
    b = []
    for l in range(r+1):
        for q in range(r+1):
            M[l].append(sum(list(map(lambda z: z**(l+q), t))))
        b.append(sum(xi*ti**l for xi, ti in zip(x, t)))
    a = gauss_slae(M, b)
    return a


def beta_pdf(x, alpha, beta):
    """
    B для байесовского статистического вывода
    """
    B = gamma(alpha)*gamma(beta) / gamma(alpha+beta)
    if x < 0 or x > 1:
        return 0
    else:
        return x**(alpha-1)*(1-x)**(beta-1)/B


def gradient(f, x, h=1e-05):
    """
    Вычисление градиента функции f в точке x

    Parameters
    ----------
        f (function): Функция
        x (list): Точка
        h (float): Шаг

    Returns
    -------
        list: Градиент
    """
    grad = []
    for i, _ in enumerate(x):
        xh = [xj+(h if j == i else 0) for j, xj in enumerate(x)]
        grad.append((f(xh)-f(x))/h)
    return grad


def grad_step(F, x, mu):
    """
    Шаг градиентного спуска

    Parameters
    ----------
        F (function): Функция
        x (list): Точка
        mu (float): Шаг

    Returns
    -------
        list: Точка
    """
    gF = gradient(F, x)
    gF = scalar_multiply(1/magnitude(gF), gF)
    return vector_add(x, scalar_multiply(-mu, gF))


def gradient_descent(F, x0, s=500, mu=0.05, gF_min=0.01, mu_min=1e-5):
    """
    Градиентный спуск

    Parameters
    ----------
        F (function): Функция
        x0 (list): Точка
        s (int): Количество шагов
        mu (float): Шаг
        gF_min (float): Минимальное значение градиента
        mu_min (float): Минимальное значение шага
    
    Returns
    -------
        list: Точка
        float: Значение функции
    """
    x = copy(x0)
    Fxn = F(x)
    for _ in range(s):
        Fx = F(x)
        muc = mu
        while Fxn >= Fx:
            xn = grad_step(F, x, mu)
            Fxn = F(xn)
            muc *= 0.5
            if muc < mu_min:
                return x, Fx
        x = xn
    return x, Fx


def minimize_stochastic(f, x, y, a_0, h_0=0.1, max_steps=1000):
    """
    Минимизация функции методом случайного поиска

    Parameters
    ----------
        f (function): Функция
        x (list): Список значений x
        y (list): Список значений y
        a_0 (list): Начальные коэффициенты
        h_0 (float): Шаг
        max_steps (int): Максимальное количество шагов

    Returns
    -------
        list: Точка
        float: Значение функции
    """
    a = a_0
    h = h_0
    min_a, min_F = None, float('inf')
    drunken_steps = 0
    while drunken_steps < max_steps:
        value = sum((f(xx, a)-yy)**2 for xx, yy in zip(x, y))
        if value < min_F:
            min_a, min_F = a, value
            drunken_steps = 0
            h = h_0
        else:
            drunken_steps += 1
            h *= 0.9
            n = random.randint(0, len(x)-1)
            grad = []
            for i, _ in enumerate(a):
                ah = a.copy()
                ah[i] += h/100
                # главное - в этом шаге (суммы нет!)
                grad.append(100*((f(x[n], ah)-y[n])**2-(f(x[n], a)-y[n])**2)/h)
            a = [a[i] - h*grad[i] for i, _ in enumerate(a)]
    return min_a, value


def negate(f):
    """
    Возвращает функцию, которая для любого входа x возвращает -f(x)

    Parameters
    ----------
        f (function): Функция

    Returns
    -------
        function: Обратная функция
    """
    return lambda *args, **kwargs: -f(*args, **kwargs)


def maximize_stochastic(f, x, y, a_0, h_0=0.01):
    """
    Максимизация функции методом случайного поиска

    Parameters
    ----------
        f (function): Функция
        x (list): Список значений x
        y (list): Список значений y
        a_0 (list): Начальные коэффициенты
        h_0 (float): Шаг

    Returns
    -------
        list: Точка
        float: Значение функции
    """
    return minimize_stochastic(negate(f), x, y, a_0, h_0)


def make_histogram(points, bucket_size):
    """
    Возвращает гистограмму

    Parameters
    ----------
        points (list): Список значений
        bucket_size (float): Размер корзины

    Returns
    -------
        dict: Гистограмма
    """
    return Counter(bucket_size * floor(point / bucket_size) for point in points)


def plot_histogram(points, bucket_size, title=''):
    """
    Рисует гистограмму

    Parameters
    ----------
        points (list): Список значений
        bucket_size (float): Размер корзины
        title (str): Заголовок
    """
    histogram = make_histogram(points, bucket_size)
    plt.bar(histogram.keys(), histogram.values(), width=bucket_size)
    plt.title(title)
    plt.show()


def random_normal():
    """
    Возвращает случайное число из нормального распределения

    Returns
    -------
        float: Случайное число
    """
    return inv_f_norm(random.random(), 0, 1)


def matrix_entry(i, j):
    return correlation(get_column(data, i), get_column(data, j))


def correlation_matrix(data):
    _, num_columns = shape(data)
    return make_matrix(num_columns, num_columns, matrix_entry)


def parse_rows_with(reader, parsers):
    for row in reader:
        yield parse_row(row, parsers)


def try_or_none(f):
    """
    Возвращает обертку, которая возвращает None, если f вызывает исключение,
    и f(x) в противном случае

    Parameters
    ----------
        f (function): Функция

    Returns
    -------
        function: Обертка
    """
    def f_or_none(x):
        try:
            return f(x)
        except:
            return None
    return f_or_none


def parse_row(input_row, parsers):
    return [try_or_none(parser)(value) if parser is not None
            else value for value, parser in zip(input_row, parsers)]


def try_parse_field(field_name, value, parser_dict):
    parser = parser_dict.get(field_name)
    if parser is not None:
        return try_or_none(parser)(value)
    else:
        return value


def parse_dict(input_dict, parser_dict):
    return {field_name: try_parse_field(field_name, value, parser_dict)
            for field_name, value in input_dict.iteritems()}


def picker(field_name):
    return lambda row: row[field_name]


def pluck(field_name, rows):
    return list(map(picker(field_name), rows))


def group_by(grouper, rows, value_transform=None):
    """
    Ключ – результат вычисления grouper, значение – список строк
    """
    grouped = defaultdict(list)
    for row in rows:
        grouped[grouper(row)].append(row)
    if value_transform is None:
        return grouped
    else:
        return {key: value_transform(rows)
                for key, rows in grouped.items()}


def scale(matrix):
    rows, cols = shape(matrix)
    means = [mean(get_column(matrix, i)) for i in range(cols)]
    stdevs = [mean(get_column(matrix, i)) for i in range(cols)]

    def res(i, j):
        if stdevs[j] > 0:
            return (matrix[i][j] - means[j])/stdevs[j]
        else:
            return matrix[i][j]
    return make_matrix(rows, cols, res)


def de_mean_matrix(A):
    """
    Возвращает результат вычитания среднего из каждого элемента
    
    Parameters
    ----------
        A (list of list): Матрица

    Returns
    -------
        list of list: Матрица
    """
    r, c = shape(A)
    columns_means = [mean(get_column(A, i)) for i in range(c)]
    return make_matrix(r, c, lambda i, j: A[i][j] - columns_means[j])


def direction(w):
    """
    Возвращает единичный вектор, указывающий в направлении w

    Parameters
    ----------
        w (list): Вектор

    Returns
    -------
        list: Единичный вектор
    """
    return [w_i/magnitude(w) for w_i in w]


def dir_variance(X, w):
    """
    Возвращает дисперсию в направлении w

    Parameters
    ----------
        X (list of list): Матрица
        w (list): Вектор

    Returns
    -------
        float: Дисперсия
    """
    return sum(dot(x_i, direction(w))**2 for x_i in X)


def the_first_priciple_comp(data):
    """
    Возвращает первую главную компоненту

    Parameters
    ----------
        data (list of list): Матрица

    Returns
    -------
        list: Вектор
    """
    n = len(data[0])
    init_w = [random.random() for _ in range(n)]
    w = gradient_descent(negate(partial(dir_variance, data)), init_w)[0]
    return direction(w)


def matrix_mul(A, B):
    """
    Возвращает результат перемножения матриц

    Parameters
    ----------
        A (list of list): Матрица
        B (list of list): Матрица

    Returns
    -------
        list of list: Матрица
    """
    C = []
    for i, ai in enumerate(A):
        if isinstance(B[0], list):
            C.append([dot(ai, get_column(B, j)) for j, _ in enumerate(B[0])])
        else:
            C.append(dot(ai, B))
    return C


def turn(data, x):
    """
    Поворачивает данные так, что ось x становится первой координатной осью

    Parameters
    ----------
        data (list of list): Матрица
        x (list): Вектор

    Returns
    -------
        list of list: Матрица
    """
    T = []
    T.append(x)
    n = len(x)
    for i in range(1, n):
        denom = sqrt(1-sum(xi**2 for xi in x[i:]))
        T.append([])
        for j, xj in enumerate(x[:i-1]):
            T[-1].append(xj*x[i]/denom)
        T[-1].append(-denom)
    new_data = []
    for d in data:
        new_data.append(matrix_mul(T, d))
    return new_data


def principal_components(data, m):
    """
    Снижает размерность данных до m методом главных компонент

    Parameters
    ----------
        data (list of list): Матрица
        m (int): Размерность

    Returns
    -------
        list of list: Матрица
    """
    res_d = [[] for _ in data]
    n = len(data[0])
    components = []
    projections = []
    for _ in range(m):
        components.append(the_first_priciple_comp(data))
        data = turn(data, components[-1])
        for j, dj in enumerate(data):
            res_d[j].append(dj[0])
        data = [di[1:] for di in data]
    return res_d


def split_data(data, prob):
    """
    Разбивает данные на две части в соотношении prob к (1 - prob)

    Parameters
    ----------
        data (list of list): Матрица
        prob (float): Вероятность

    Returns
    -------
        list of list: Матрица
    """
    results = [], []
    for row in data:
        results[0 if random.random() < prob else 1].append(row)
    return results


def train_test_split(x, y, test_pct):
    """
    Разбивает данные на тренировочные и тестовые

    Parameters
    ----------
        x (list): Вектор x
        y (list): Вектор y
        test_pct (float): Вероятность

    Returns
    -------
        list: Вектор x тренировочный
        list: Вектор x тестовый
        list: Вектор y тренировочный
        list: Вектор y тестовый
    """
    data = zip(x, y)
    train, test = split_data(data, 1 - test_pct)
    x_train, y_train = zip(*train)
    x_test, y_test = zip(*test)
    return x_train, x_test, y_train, y_test


def precision(true_pos, false_pos):
    """
    Возвращает точность

    Parameters
    ----------
        true_pos (int): Количество истинно положительных
        false_pos (int): Количество ложно положительных

    Returns
    -------
        float: Точность
    """
    return true_pos/(true_pos + false_pos)


def recall(true_pos, false_neg):
    """
    Возвращает полноту

    Parameters
    ----------
        true_pos (int): Количество истинно положительных
        false_neg (int): Количество ложно отрицательных

    Returns
    -------
        float: Полнота
    """
    return true_pos/(true_pos + false_neg)


def f1_score(true_pos, false_pos, false_neg):
    """
    Возвращает F1 метрику на истинно положительных, ложно положительных 
    и ложно отрицательных

    Parameters
    ----------
        true_pos (int): Количество истинно положительных
        false_pos (int): Количество ложно положительных
        false_neg (int): Количество ложно отрицательных

    Returns
    -------
        float: F1 метрика
    """
    p = precision(true_pos, false_pos)
    r = recall(true_pos, false_neg)
    return 2*p*r/(p+r)


def majority_vote(labels):
    """
    Возвращает наиболее частую метку

    Parameters
    ----------
        labels (list): Метки, упорядочены от ближней к дальней

    Returns
    -------
        int: Метка
    """
    vote_counts = Counter(labels)
    winner, winner_count = vote_counts.most_common(1)[0]
    num_winners = len([count for count in vote_counts.values()
                       if count == winner_count])
    if num_winners == 1:
        return winner
    else:
        return majority_vote(labels[:-1])  # пытаемся снова


def knn_classify(k, labeled_points, new_point):
    """
    Классификация методом k ближайших соседей

    Parameters
    ----------
        k (int): Количество соседей
        labeled_points (list): Метки
        new_point (list): Новая точка

    Returns
    -------
        int: Метка
    """
    # сначала сортируем
    by_distance = sorted(labeled_points,
                         key=lambda point: distance(point[0], new_point))
    # k ближайших находим
    k_nearest_labels = [label for _, label in by_distance[:k]]
    # голосуем и пробуем снова
    return majority_vote(k_nearest_labels)

def k_neighbours_classify(k, data):
    """
    Классификация методом k ближайших соседей. Возвращает словарь с результатами
    для k равного 1, 2, ..., k 

    Parameters
    ----------
        k (int): Количество соседей
        data (list): Метки
    
    Returns
    -------
        dict: Словарь с результатами (количество правильных, общее количество)
    
    Author: @Kseen715
    """
    result = {}
    for k in range(1, k + 1):
        n_correct = 0
        for item in data:
            values, actual_br = item
            other_items = [other_item for other_item in data \
                if other_item != item]
            predicted_br = knn_classify(k, other_items, values)
            if predicted_br == actual_br:
                n_correct += 1
        result[k] = (n_correct, len(data))
    return result

def tokenize(message):
    """
    Разбивает сообщение на уникальные слова

    Parameters
    ----------
        message (str): Сообщение

    Returns
    -------
        set: Уникальные слова
    """
    message = message.lower()  # нам не нужны большие буквы
    # извлекаем слова
    all_words = re.findall("[a-z0-9']+", message)
    # возвращаем только уникальные
    return set(all_words)


def count_words(training_set):
    """
    Считает количество слов в сообщениях

    Parameters
    ----------
        training_set (list): Тренировочный набор

    Returns
    -------
        dict: Словарь с количеством слов
    """
    counts = defaultdict(lambda: [0, 0])
    for message, is_spam in training_set:
        for word in tokenize(message):
            counts[word][0 if is_spam else 1] += 1
    return counts


def word_probabilities(counts, total_spams, total_non_spams, k=0.5):
    return [(w, (spam + k) / (total_spams + 2*k),
             (non_spam + k) / (total_non_spams + 2*k))
            for w, (spam, non_spam) in counts.iteritems()]


def spam_probability(word_probs, message):
    message_words = tokenize(message)
    lp_spam = lp_not_spam = 0.0
    for word, p_spam, p_not_spam in word_probs:
        if word in message_words:
            lp_spam += log(p_spam)
            lp_not_spam += log(p_not_spam)
        else:
            lp_spam += log(1 - p_spam)
            lp_not_spam += log(1 - p_not_spam)
    ps = exp(lp_spam)
    pns = exp(lp_not_spam)
    return ps/(ps+pns)


def predict(alpha, beta, x_i):
    """
    Уравнение прямой в точке x_i

    Parameters
    ----------
        alpha (float): Смещение
        beta (float): Угол наклона
        x_i (float): Точка

    Returns
    -------
        float: Значение y
    """
    return beta*x_i + alpha


def error(alpha, beta, x_i, y_i):
    """
    Ошибка на i-м отсчете

    Parameters
    ----------
        alpha (float): Смещение
        beta (float): Угол наклона
        x_i (float): Точка
        y_i (float): Значение

    Returns
    -------
        float: Ошибка
    """
    return y_i - predict(alpha, beta, x_i)


def least_squares_fit(x, y):
    """
    Определение коэффициентов простой линейной регрессии

    Parameters
    ----------
        x (list): Массив x
        y (list): Массив y

    Returns
    -------
        float: Смещение
        float: Угол наклона
    """
    beta = correlation(x, y)*standard_deviation(y) / standard_deviation(x)
    alpha = mean(y) - beta*mean(x)
    return alpha, beta


def total_sum_of_squares(y):
    """
    Полная сумма квадратов отклонений от среднего

    Parameters
    ----------
        y (list): Массив значений

    Returns
    -------
        float: Сумма квадратов
    """
    return sum(ym_i**2 for ym_i in de_mean(y))


def sum_of_squared_errors(alpha, beta, x, y):
    return sum(error(alpha, beta, x_i, y_i) ** 2 for x_i, y_i in zip(x, y))


def r_squared(alpha, beta, x, y):
    """
    Коэффициент детерминации (R-квадрат)

    Parameters
    ----------
        alpha (float): Смещение
        beta (float): Угол наклона
        x (list): Массив значений x
        y (list): Массив значений y

    Returns
    -------
        float: Коэффициент детерминации (R-квадрат)
    """
    return (1 - sum_of_squared_errors(alpha, beta,
                                      x, y) / total_sum_of_squares(y))


def regression(X, y):
    n = len(y)
    M, b = [], []
    M.append([sum(x) for x in X]+[n])
    b.append(sum(y))
    for l, xl in enumerate(X):
        M.append([dot(x, xl) for x in X]+[sum(xl)])
        b.append(dot(y, xl))
    beta = gauss_slae(M, b)
    return beta  # beta[-1],beta[:-1]


def mult_predict(x_i, beta):
    return sum(xij*bj for xij, bj in zip(x_i, beta))+beta[-1]


def mult_error(x_i, y_i, beta):
    return mult_predict(x_i, beta) - y_i


def mult_r_squared(x, y, beta):
    sum_of_squared_errors = sum(mult_error(x_i, y_i, beta)**2
                                for x_i, y_i in zip(x, y))
    return 1.0 - sum_of_squared_errors / total_sum_of_squares(y)


def bootstrap(x, n, stats='normal'):
    if stats == 'normal':
        return [random.gauss(mean(x), standard_deviation(x)) for _ in range(n)]
    elif stats == 'even':
        return [min(x) + random.random()*(max(x) - min(x)) for _ in range(n)]
    else:
        return None


def sigmoid(x):
    return 1/(1+exp(-x))


def sigmoid_d(x):
    return sigmoid(x)*(1-sigmoid(x))


def log_likelyhood_i(x_i, y_i, beta):
    if y_i == 1:
        return log(sigmoid(dot(x_i, beta)))
    else:
        return log(1 - sigmoid(dot(x_i, beta)))


def log_likelyhood(x, y, beta):
    return sum(log_likelyhood_i(x_i, y_i, beta) for x_i, y_i in zip(x, y))


def log_partial_ij(x_i, y_i, beta, j):
    return (y_i - sigmoid(dot(x_i, beta)))*x_i[j]


def log_grad_i(x_i, y_i, beta):
    return [log_partial_ij(x_i, y_i, beta, j) for j, _ in enumerate(beta)]


def log_grad(x, y, beta):
    return reduce(vector_add,
                  [log_grad_i(x_i, y_i, beta) for x_i, y_i in zip(x, y)])


def entropy(class_probabilities):
    return sum(-p*math.log(p, 2) for p in class_probabilities if p)


def class_probabilities(labels):
    total_count = len(labels)
    return [count / total_count for count in Counter(labels).values()]


def data_entropy(labeled_data):
    labels = [label for _, label in labeled_data]
    probabilities = class_probabilities(labels)
    return entropy(probabilities)


def partition_entropy(subsets):
    total_count = sum(len(subset) for subset in subsets)
    return sum(data_entropy(subset) *
               len(subset) / total_count for subset in subsets)


def partition_by(inputs, attribute):
    # разбиение входящих данных по атрибуту
    groups = defaultdict(list)
    for inp in inputs:
        key = inp[0][attribute]
        groups[key].append(inp)
    return groups


def partition_entropy_by(inputs, attribute):
    # энтропия разбиения входящих данных по атрибуту
    partitions = partition_by(inputs, attribute)
    return partition_entropy(partitions.values())


def build_tree_id3(inputs, split_candidates=None):
    # построим дерево на ID3
    if split_candidates is None:
        split_candidates = inputs[0][0].keys()
    num_inputs = len(inputs)
    num_trues = len([label for item, label in inputs if label])
    num_falses = num_inputs - num_trues
    if num_trues == 0:
        return False
    if num_falses == 0:
        return True
    if not split_candidates:
        return num_trues >= num_falses
    best_attribute = min(split_candidates,
                         key=partial(partition_entropy_by, inputs))
    partitions = partition_by(inputs, best_attribute)
    new_candidates = [a for a in split_candidates if a != best_attribute]
    subtrees = {attribute_value: build_tree_id3(subset, new_candidates)
                for attribute_value, subset in iter(partitions.items())}
    subtrees[None] = num_trues > num_falses
    return (best_attribute, subtrees)


def classify(tree, inp):
    # алгоритм классификации входящих значений по дереву принятия решений
    if tree in (True, False):
        return tree
    attribute, subtree_dict = tree
    subtree_key = inp.get(attribute)
    if subtree_key not in subtree_dict:
        subtree_key = None
    subtree = subtree_dict[subtree_key]
    return classify(subtree, inp)


def fwd(x, w, b):  
    # прямой проход сигналом x одного нейрона с весами w и смещением b
    return sigmoid(dot(w, x)+b)


def ch_weights(w, b, data, mu=0.05):
    # data есть список списков с меткой в конце каждого вектора
    X, l = [di[:-1] for di in data], [di[-1] for di in data]
    y = [fwd(x, w, b) for x in X]
    new_w = []
    for k, _ in enumerate(X[0]):
        new_w.append(w[k] - mu*sum((y[a]-l[a])*y[a]*(1-y[a])*X[a][k]
                                   for a, _ in enumerate(X)))
    b -= mu*sum((ya-la)*ya*(1-ya) for ya, la in zip(y, l))
    return new_w, b


def art_nn(entry, layers):  
    # инициирует веса и смещения для указанного числа входов и списка из 
    # количеств нейронов в слоях
    WW, W, B, b = [], [], [], []
    for li in range(layers[0]):
        W.append([random.random()-1/2 for _ in range(entry)])
        b.append(0)
    WW.append(W)
    B.append(b)
    if len(layers) == 1:
        return WW, B
    else:
        for s, ls in enumerate(layers[1:]):
            W, b = [], []
            for lj in range(ls):
                W.append([random.random()-1/2 for _ in range(layers[s])])
                b.append(0)
            WW.append(W)
            B.append(b)
        return WW, B


def forward(data, WW, B):  # data не содержат метки
    layer_outputs = [[] for _ in B]
    for d in data:
        oi = []
        for i, bi in enumerate(B[0]):
            oi.append(fwd(d, WW[0][i], bi))
        layer_outputs[0].append(oi)
        if len(B) == 1:
            continue
        else:
            k = 1
            for W, b in zip(WW[1:], B[1:]):
                oo = []
                for i, bi in enumerate(b):
                    oo.append(fwd(oi, W[i], bi))
                oi = copy(oo)
                layer_outputs[k].append(oo)
                k += 1
    return layer_outputs


# labeled_data содержит номер возбужденного нейрона в конце каждого вектора
def layer_train(labeled_data, W, b, mu=0.05):
    Y, L, Delta = [], [], []
    Y = forward([ld[:-1] for ld in labeled_data], [W], [b])[-1]
    for ld in labeled_data:
        L.append([1 if ld[-1] == i else 0 for i, _ in enumerate(b)])
        Delta.append([(Y[-1][i]-L[-1][i])*Y[-1][i]*(1-Y[-1][i])
                      for i, _ in enumerate(b)])
    new_W = [[wk - mu*sum(Delta[a][i]*labeled_data[a][k]
                          for a, _ in enumerate(L))
              for k, wk in enumerate(w)]
             for i, w in enumerate(W)]
    new_b = [bi - mu*sum(Delta[a][i] for a, _ in enumerate(L))
             for i, bi in enumerate(b)]
    return new_W, new_b, Delta


def train_step(labeled_data, WW, B, mu=0.05):
    new_WW, new_B = [[] for _ in B], [[] for _ in B]
    data = [ld[:-1] for ld in labeled_data]
    layer_outputs = forward(data, WW, B)
    n = len(B) - 1
    Delta = 0
    if n == 0:
        new_WW[n], new_B[n] = layer_train(labeled_data, WW[-1], B[-1], mu)[:-1]
    else:
        new_WW[n], new_B[n], Delta = \
            layer_train(layer_outputs[-2], WW[-1], B[-1], mu)
        for i in range(1, n+1):
            Z = layer_outputs[n-i]
            H = [[Z[a][j] - sum(Delta[a][k]*WW[n-i+1][k][j]
                                for k, _ in enumerate(B[n-i+1]))
                 for j, _ in enumerate(B[n-i])] for a, _ in enumerate(Z)]
            LD = [z+h for z, h in zip(Z, H)]
            new_WW[n-i], new_B[n -
                               i], Delta = layer_train(LD, WW[n-i], B[n-i], mu)
    return new_WW, new_B


def volume(data):
    return [min([di[j] for di in data]) for j, _
            in enumerate(data[0])], [max([di[j] for di in data])
                                     for j, _ in enumerate(data[0])]


def rand_centers(V, k):
    centers = defaultdict(list)
    n = len(V[0])
    for i in range(k):
        for j in range(n):
            centers[i].append(random.randint(V[0][j], V[1][j]))
    return centers


def group(C, data):
    clasters = defaultdict(list)
    for d in data:
        d_inf = float('inf')
        for j in C.keys():
            if distance(d, C[j]) < d_inf:
                i = j
                d_inf = distance(d, C[j])
        clasters[i].append(d)
    return clasters


def step_centers(G):
    new_C = defaultdict(list)
    for k in G.keys():
        n = len(G[k][0])
        for i in range(n):
            new_C[k].append(sum(g[i] for g in G[k])/n)
    return new_C


def compare(C1, C2, m=2):
    # m - это число знаков после запятой,  
    # принимаемых во внимание
    check_list = []
    for k in C1.keys():
        check_list.append(round(sum((c1-c2)**2
                                    for c1, c2 in zip(C1[k], C2[k])), m))
    return any(check_list)


def clustering(data, k):
    """
    Кластеризация методом k-средних

    Parameters
    ----------
        data (list of lists): Данные, которые нужно кластеризовать
        k (int): Число кластеров

    Returns
    -------
        new_C (dict): Центры кластеров
        G (dict): Кластеры
    """
    V = volume(data)
    C = rand_centers(V, k)
    G = group(C, data)
    new_C = step_centers(G)
    while compare(C, new_C):
        C = dict(new_C)
        G = group(C, data)
        new_C = step_centers(G)
    return dict(new_C), dict(G)


class KMeans:
    """
    Кластеризация методом k-средних
    """
    def __init__(self, k):
        """
        Parameters
        ----------
            k (int): Число кластеров
        """
        self.k = k
        self.means = None

    def classify(self, inp):
        """
        Parameters
        ----------
            inp (list): Входные данные 

        Returns
        -------
            int: Номер кластера
        """
        return min(range(self, k), flag=lambda i:
                   squared_distance(inp, self.means[i]))

    def train(self, inps):
        """
        Кластеризация данных

        Parameters
        ----------
            inps (list of lists): Данные, которые нужно кластеризовать
        """
        self.means = random.sample(inps, self.k)
        assignments = None
        while True:
            new_assignments = map(self.classify, inps)
            if assignments == new_assignments:
                return
            assignments = new_assignments
            for i in range(self.k):
                i_points = [p for p, a in zip(inps, assignments) if a == i]
                if i_points:
                    self.means[i] = vector_mean(i_points)


def squared_errors(inps, k):
    clasterbuilder = KMeans(k)
    clasterbuilder.train(inps)
    means = clasterbuilder.means
    inclaster = map(clasterbuilder.classify, inps)
    return sum(squared_distance(inp, means[claster])
               for inp, cluster in zip(inps, inclaster))


def is_leaf(cluster):
    return len(cluster) == 1


def get_children(cluster):
    if is_leaf(cluster):
        raise TypeError('листовой не имеет дочерних')
    else:
        return cluster[1]


def get_values(cluster):
    if is_leaf(cluster):
        return cluster
    else:
        return [val for val in get_values(child)
                for child in get_children(cluster)]


def cluster_distance(cluster1, cluster2, distance_agg=min):
    return distance_agg([distance(inp1, inp2)
                         for inp1 in get_values(cluster1) \
                            for inp2 in get_values(cluster2)])


def get_merge_order(cluster):
    if is_leaf(cluster):
        return float('inf')
    else:
        return cluster[0]


def bottom_up_cluster(inps, distance_agg=min):
    clusters = [(inp,) for inp in inps]
    while len(clusters) > 1:
        c1, c2 = min([(cluster1, cluster2)
                      for i, cluster1 in enumerate(clusters)
                      for cluster2 in clusters[:i]],
                     key=lambda x, y: cluster_distance(x, y, distance_agg))
        clusters = [c for c in clusters if c != c1 and c != c2]
        merged_cluster = (len(clusters), [c1, c2])
        clusters.append(merged_cluster)
    return clusters[0]


def generate_clusters(base_cluster, num_clusters):
    clusters = [base_cluster]
    while len(clusters) < num_clusters:
        next_cluster = min(clusters, key=get_merge_order)
        clusters = [c for c in clusters if c != next_cluster]
        clusters.extend(get_children(next_cluster))
    return clusters


def trigram_writer(url, tag, attr_key_value):
    html = requests.get(url).text
    soup = BS(html, 'html5lib')
    content = soup.find(tag, exec(attr_key_value))
    regex = r"[\w']+|[\.]"
    words = re.findall(regex, content.text)
    trigrams = zip(words, words[1:], words[2:])
    trans = defaultdict(list)
    init = []
    for first, second, third in trigrams:
        if first == '.':
            init.append(second)
        trans[(first, second)].append(third)
    second = random.choice(init)
    first = '.'
    result = [second]
    while True:
        candidates = trans[(first, second)]
        next_word = random.choice(candidates)
        first, second = second, next_word
        result.append(second)
        if second == '.':
            return " ".join(result)


def gs_step_down(data, powers={}):  # just non-negative integers in data
    pwrs = getpower(data, powers)
    smallest = dict(sorted(pwrs.items(), key=lambda item: item[1])[:2])
    n1, n2 = tuple(smallest.keys())
    newdata = []
    for d in data:
        nd = [di if i not in [n1, n2] else d[n1] + smallest[n1]*d[n2]
              for i, di in enumerate(d) if i != n2]
        newdata.append(nd)
    # features number n1 and n2 are unified into n1, number n2 is dropped 
    # in newdata
    return newdata, n1, n2, smallest[n1], smallest[n2]


# performs backward from down transformation of the newdata to data
def gs_step_downback(newdata, n1, n2, N1):
    data = []
    for nd in newdata:
        d = [ndi if i != n1 else ndi % N1 for i, ndi
             in enumerate(nd[:n2])] + [nd[n1]//N1] + [ndi if i != n1-1 \
                else ndi % N1 for i, ndi in enumerate(nd[n2:])]
        data.append(d)
    return data


def gs_step_up(data, powers={}):  # just non-negative integers in data
    pwrs = getpower(data, powers)
    biggest = dict(sorted(pwrs.items(), key=lambda item: item[1],
                          reverse=True)[:1])
    n1 = list(biggest.keys())[0]
    B = biggest[n1]
    N1 = int(B**(1/2))
    N2 = (B//N1 if not B % N1 else B//N1 + 1)
    newdata = []
    for d in data:
        nd = (d[:n1] + [d[n1] % N1] + d[n1+1:] + [d[n1]//N1]) \
            if n1 < len(d)-1 else (d[:n1] + [d[n1] % N1] + [d[n1]//N1])
        newdata.append(nd)
    # feature number n1 is splited onto feature number n1 with power N1 and the 
    # last feature powered N2 in newdata
    return newdata, n1, N1, N2


# performs backward from up transformation of the newdata to data
def gs_step_upback(newdata, n1, N1):
    n = len(newdata[0]) - 1
    data = []
    for nd in newdata:
        d = [ndi if i != n1 else ndi+N1*nd[-1]
             for i, ndi in enumerate(nd[:-1])]
        data.append(d)
    return data


def getpower(data, powers={}):
    n = len(data[0])
    for k in range(n):
        if k in powers.keys():
            continue
        else:
            powers[k] = max([d[k] for d in data])+1
    return powers


def genser(data, m, powers={}):
    """
    Обобщенная сериализация

    Parameters
    ----------
        data (list of lists): Данные для сериализации
        m (int): Размерность новых данных
        powers (dict): Словарь степеней для каждого признака

    Returns
    -------
        newdata (list of lists): Новые данные
        stat_dict (dict): Словарь с параметрами для обратного преобразования
    """
    stat_dict = {}
    n = len(data[0])
    newdata = copy(data)
    pwrs = dict(powers)
    if m < n:
        for k in range(n-m):
            newdata, n1, n2, N1, N2 = gs_step_down(newdata, pwrs)
            stat_dict[-k-1] = [n1, n2, N1, N2]
            pwrs[n1] = N1*N2
            del pwrs[n2]
            pwrs = getpower(newdata, pwrs)
    elif m > n:
        for k in range(m-n):
            newdata, n1, N1, N2 = gs_step_up(newdata, pwrs)
            stat_dict[k+1] = [n1, N1, N2]
            pwrs[n1] = N1
            pwrs[len(pwrs)] = N2
            pwrs = getpower(newdata, pwrs)
    return newdata, stat_dict
