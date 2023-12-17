import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from scipy.stats import norm, uniform


def plot_phi(x_k, theta, eps, m) -> None:
    '''
    inputs:
        - x_k: np.array[flota] массив размерностью (n)- признаки модели
        - theta: np.array[float] массив размерностью (m + 1)- веса модели 
        - eps: np.array[float] массив с размерностью (n)- шум модели
        - m: int- Сложность модели (степень многочлена)  
    '''
    x = np.linspace(-4, 4, 100)
    X = np.vander(x, m + 1, increasing=True)
    X_k = np.vander(x_k, m + 1, increasing=True)
    plt.figure()
    plt.title('Истинный полезный сигнал и набор наблюдений')
    plt.plot(x, np.dot(X, theta), color='green', label='Истинный полезный сигнал')
    plt.scatter(x_k, np.dot(X_k, theta) + eps, color='black', label='Набор наблюдений')
    plt.legend(loc="upper left", fontsize=10)
    plt.xlabel('X')
    plt.ylabel('Values')
    plt.grid()
    plt.show()

def plot_phi_estimation(theta, w, m) -> None:
    '''
    inputs:
        - theta: np.array[float] массив размерностью (m + 1)- истинные веса модели 
        - w: np.array[float] массив с размерностью (m + 1)- найденные веса модели
        - m: int- Сложность модели (степень многочлена)  
    '''
    x = np.linspace(-4, 4, 100)
    X = np.vander(x, m + 1, increasing=True)
    plt.figure()
    plt.title('Полезный сигнал и оценка полезного сигнала')
    plt.plot(x, np.dot(X, theta), color='green', label='Истинный полезный сигнал')
    plt.plot(x, np.dot(X, w), color='black', label='Оценка полезного сигнала')
    plt.legend(loc="upper left", fontsize=10)
    plt.xlabel('X')
    plt.ylabel('Values')
    plt.grid()
    plt.show()

def plot_phi_intervals(theta, w, m, limits) -> None:
    '''
    inputs:
        - theta: np.array[float] массив размерностью (m + 1)- истинные веса модели 
        - w: np.array[float] массив с размерностью (m + 1)- найденные веса модели
        - m: int- Сложность модели (степень многочлена) 
        - limits: dict- значения доверительных интревалов уровней доверия 95 и 99 
    '''
    x = np.linspace(-4, 4, 100)
    X = np.vander(x, m + 1, increasing=True)
    symb_x = sp.symbols('x')
    fig, ax = plt.subplots(2, 1, figsize=(10, 12))
    for i, alpha in enumerate(limits.keys()):
        l, r = limits[alpha]
        l, r = sp.lambdify(symb_x, l), sp.lambdify(symb_x, r)
        
        ax[i].set_title(f'Уровень доверия {alpha}')
        ax[i].plot(x, np.dot(X, theta), color='green', label='Истинный полезный сигнал')
        ax[i].plot(x, np.dot(X, w), color='black', label='Оценка полезного сигнала')
        ax[i].fill_between(x, l(x), r(x), color='#7d7878', label='Доверительный интревал')
        ax[i].legend(loc="upper left", fontsize=10)
        ax[i].set_xlabel('X')
        ax[i].set_ylabel('Values')
        ax[i].grid()
        
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle('Доверительные интервалы полезного сигнала')


def plot_hist(y, y_pred, n) -> None:
    '''
    inputs:
        - y: np.array[float] массив размерностью (n)- истинные зависимые переменные(target) 
        - y_pred: np.array[float] массив размерностью (n)- найденные зависимые переменные(prediction) 
        - n: int- Размер выборки (количество объектов)
    '''
    diff = y - y_pred
    num_cols: int = int(3.322 * np.log10(n)) + 1
    _, bins = np.histogram(y - y_pred, bins=num_cols, density=True)
    values: np.array = np.linspace(bins[0], bins[-1], 100)
    distribution = norm.pdf(values, 0, np.std(diff))
    plt.figure()
    plt.title('Гистограма плотности распределения случайной ошибки')
    plt.hist(diff, bins=num_cols, density=True, color='#7d7878', edgecolor='black')
    plt.plot(values, distribution, color='green', label='Распределение')
    plt.legend(loc="upper left", fontsize=10)
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.show()
