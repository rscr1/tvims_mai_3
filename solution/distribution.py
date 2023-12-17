import numpy as np

class Distribution:
    '''
    Класс Распределения 
    inputs:
        - theta: np.array[float] массив размерностью (m + 1)- веса модели 
        - var: float- Стандартное отклонение 
        - n: int- Размер выборки (количество объектов)
        - m: int- Сложность модели (степень многочлена)
        - normal_distribution: bool- Вид распределения (True - Normal, False - Uniform)
        - seed: int- Воспроизводимость случайных результатов
    '''
    def __init__(self, theta: np.array, var: float, n: int, m: int, normal_distribution: bool, seed: int) -> None:
        self.theta = theta[:m+1]
        self.var = var
        self.n = n
        self.m = m
        self.normal_distribution = normal_distribution
        self.seed = seed
        
        self.eps = self.gen_distribution()
        self.x_k = self.calculate_x_k()
        self.X = self.calculate_X()
        self.phi = self.calculate_phi()
        self.y = self.calculate_y()

        self.w = self.fit()
        self.y_pred = self.predict()
        self.mse = self.calculate_mse()
        self.se = self.calculate_se()


    def gen_distribution(self) -> np.array:
        '''
        Функция для генерации распределения шума
        returns: np.array[float] массив с размерностью (n)- неустранимый шум
        '''
        np.random.seed(seed=self.seed)
        if self.normal_distribution:
            eps: np.array = np.random.normal(0, self.var, self.n)
        else:
            eps: np.array = np.random.uniform(-3 * np.sqrt(self.var), 3 * np.sqrt(self.var), self.n)
        return eps

    def calculate_x_k(self) -> np.array:
        '''
        Функция для рассчета столбца признака
        returns: np.array[float] массив с размерностью (n)- признак
        '''
        return np.array([-4 + (k + 1) * 8 / self.n for k in range(self.n)])

    def calculate_X(self) -> np.array:
        '''
        Функция для рассчета матрицы объект-признак
        returns: np.array[np.array[float]] массив с размерностью (n, m + 1)- матрица объект-признак
        '''    
        return np.vander(self.x_k, self.m + 1, increasing=True)

    def calculate_phi(self) -> np.array:
        '''
        Функция для рассчета полезного сигнала
        returns: np.array[float] массив с размерностью (n)- полезный сигнал
        '''
        return np.dot(self.X, self.theta)

    def calculate_y(self) -> np.array:
        '''
        Функция для рассчета зависимой переменной y
        returns: np.array[float] массив с размерностью (n)- зависимая переменная
        '''
        return self.phi + self.eps

    def fit(self) -> np.array:
        '''
        Функция поиска параметров модели
        returns: np.array[float] массив с размерностью (n)- веса модели
        '''
        self.w = np.dot(np.dot(np.linalg.inv(np.dot(self.X.T, self.X)), self.X.T), self.y)
        return self.w

    def predict(self) -> np.array:
        '''
        Функция предсказания зависимой переменной
        returns: np.array[float] массив с размерностью (n)- зависимая переменная
        '''
        return np.dot(self.X, self.w) 

    def calculate_mse(self) -> float:
        '''
        Функция для рассчета Среднеквадратичного отклонения
        returns: float число- Среднеквадратичное отклонение
        '''
        return (((self.y - self.y_pred))**2 / (self.n - self.m - 1)).sum()

    def calculate_se(self) -> float:
        '''
        Функция для рассчета Среднеквадратичного отклонения
        returns: float число- Среднеквадратичное отклонение
        '''
        return np.sqrt(np.diag(self.calculate_mse() * np.linalg.inv(np.dot(self.X.T, self.X))))
