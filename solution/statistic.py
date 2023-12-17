import numpy as np
import sympy as sp

from scipy.stats import t, norm, chi2


def t_test(x:np.array, y: np.array, n: int, alpha: float) -> bool:
    '''
    Функция для рассчета степени многочлена полезного сигнала
    inputs:
        - x: np.array[float]: массив с размерностью (m + 1)- веса модели
        - se: float- Стандартная ошибка
        - n: int- Размер выборки (количество объектов)
        - m: int- Сложность модели (степень многочлена)
        - alpha: float- Уровень значимости
    returns: bool- Является ли m степенью многочлена.
    '''
    X: np.array = np.vander(x, n + 1, increasing=True) 
    num_samples, num_features = X.shape
    w: np.array = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
    mse: float = ((y - np.dot(X, w))**2 / (num_samples - num_features)).sum()
    se: np.array = np.sqrt(np.diag(mse * np.linalg.inv(np.dot(X.T, X))))
    T: np.array = np.abs(w / se)
    df: int = num_samples - num_features
    p_value = 2 * (1 - t.cdf(T, df=df))
    if p_value[n] < alpha:
        print(f'Степень многочлена = {n} является статистически значимым.')
        return True
    print(f'Степень многочлена = {n} не является статистически значимым.\nСтепень многочлена равна {n - 1}')
    return False

def calculate_confidence_interval_w(se: float, w: float, n: int, m: int, alpha: float) -> tuple:
    '''
    Функция для рассчета доверительных интервалов весов
    inputs:
        - se: float- Стандартная ошибка
        - w: np.array[float] массив с размерность (m + 1)- Веса модели
        - n: int- Размер выборки (количество объектов)
        - m: int- Сложность модели (степень многочлена)
        - alpha: float- Уровень значимости
    returns: tuple массив с размерностью(2)- Доверительный интервал
    '''
    t_ = t.ppf(1 - (1 - alpha) / 2, df=n - m - 1)
    return (w + i * se * t_ for i in [-1, 1])

def calculate_confidence_interval_phi(mse: float, X: np.array, w: np.array, n: int, m: int, alpha: float) -> tuple:
    '''
    Функция для рассчета доверительных интервалов полезного сигнала
    inputs:
        - mse: float- Среднеквадратичная ошибка
        - X: np.array[np.array[float]] массив с размерностью(n, m + 1)- Матрица объект-признак
        - w: np.array[float] массив с размерность (m + 1)- Веса модели
        - n: int- Размер выборки (количество объектов)
        - m: int- Сложность модели (степень многочлена)
        - alpha: float- Уровень значимости
    returns: tuple массив с размерностью(2)- Доверительный интервал
    '''
    symb_x = sp.symbols("x")
    x = sp.Matrix(np.vander(np.array([symb_x]), m + 1, increasing=True))
    t_ = t.ppf(1 - (1 - alpha) / 2, df=n - m - 1)
    root = sp.sqrt(mse * (x * np.linalg.inv(np.dot(X.T, X)) * x.T))
    f_pred = x * sp.Matrix(w)
    return (f_pred + i *  root * t_ for i in [-1, 1])

def chi2_test(x, y, y_pred, n) -> tuple:
    diff = y - y_pred
    num_bins = int(3.32 * np.log10(n)) + 1
    hist, bin_edges = np.histogram(diff, bins=num_bins, density=True)
    bin_width = bin_edges[1] - bin_edges[0]
    expected_freq = np.diff([0]+ list(norm.cdf(bin_edges, loc=0, scale=(np.sum(diff**2) / n)))+[1])
    estimated_freq = [0] + list(hist * bin_width) + [0]
    chi2_ = len(x) * np.sum((estimated_freq - expected_freq)**2 / expected_freq)
    p_value = 1 - np.sum(chi2.cdf(chi2_, df=num_bins - 1))
    return p_value, chi2_
