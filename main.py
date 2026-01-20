# venv
# Python 3.12.9

import numpy as np
import random
from cya_interpolator import ColumnInterpolator, RowInterpolator

# Выводить ли в консоль информацию о текущих действиях в программе
write_progress = True

# Таблица "Коэффициент подъёмной силы Cya" из пособия
table_alphas = [-2, 0, 2, 4, 6, 8, 10]
table_machs = [0.7, 0.9, 1.1, 1.4, 2, 3, 4, 6]
table_cya = [
    [0, 0, 0, 0, 0, -0.0062, -0.0097, -0.0118],
    [0.0437, 0.0452, 0.0531, 0.0683, 0.0620, 0.0127, -0.006, -0.002],
    [0.0874, 0.0904, 0.1062, 0.1366, 0.1239, 0.0543, 0.0219, 0.0159],
    [0.1311, 0.1356, 0.1593, 0.2044, 0.1858, 0.0926, 0.0493, 0.035],
    [0.1748, 0.1818, 0.2124, 0.2732, 0.2478, 0.1401, 0.0866, 0.0601],
    [0.2185, 0.2260, 0.2655, 0.3415, 0.3097, 0.1832, 0.1204, 0.0888],
    [0.2622, 0.2712, 0.3186, 0.4097, 0.3717, 0.2298, 0.1583, 0.1193]
]

# Списки объектов, хранящие вспомогательные данные для интерполяции
# Например чтобы не решать СЛАУ 28 * 28 в каждой точке
interpolated_rows = []
interpolated_columns = []

for i in range(len(table_machs)):
    if write_progress:
        print("Интерполируем столбец " + str(i + 1))

    _values = []

    for row_number in range(len(table_alphas)):
        _values.append(table_cya[row_number][i])

    interpolated_columns.append(ColumnInterpolator(table_alphas, _values))

for i in range(len(table_alphas)):
    if write_progress:
        print("Интерполируем строку " + str(i + 1))

    interpolated_rows.append(RowInterpolator(table_machs, table_cya[i]))


def cya_evaluate(_alpha, _mach):
    global table_alphas, table_machs, table_cya

    if _alpha in table_alphas and _mach in table_machs:
        return table_cya[table_alphas.index(_alpha)][table_machs.index(_mach)]

    if _mach in table_machs:
        return cya_interpolate_alpha(_alpha, table_machs.index(_mach))

    if _alpha in table_alphas:
        return cya_interpolate_mach(table_alphas.index(_alpha), _mach)

    return cya_interpolate_alpha_mach_using_rows(_alpha, _mach)


def cya_interpolate_alpha(_alpha, _index):
    global interpolated_columns

    return interpolated_columns[_index].evaluate(_alpha)


def cya_interpolate_mach(_index, _mach):
    global interpolated_rows

    return interpolated_rows[_index].evaluate(_mach)


def cya_interpolate_alpha_mach_using_rows(_alpha, _mach):
    global interpolated_rows, table_alphas

    _cya = []

    for element in interpolated_rows:
        _cya.append(element.evaluate(_mach))

    _column_interpolator = ColumnInterpolator(table_alphas, _cya)

    return _column_interpolator.evaluate(_alpha)


alphas = []

for alpha in np.linspace(table_alphas[0], table_alphas[-1], 1000):
    alphas.append(alpha)

machs = []

for mach in np.linspace(table_machs[0], table_machs[-1], 1000):
    machs.append(mach)

cya = []

if write_progress:
    print("Начало вычисления функции в точках")

for i in range(len(alphas)):
    cya.append([])

    if write_progress:
        progress = i * len(machs) / (len(alphas) * len(machs)) * 100
        print("Вычисление функции в точках, прогресс: " + str(progress) + " %")

    for j in range(len(machs)):
        cya[i].append(cya_evaluate(alphas[i], machs[j]))

# Для самопроверки выведем несколько рандомных значений. Они должны быть в пределах значений из таблицы
# в соответствующих 4 соседних ячейках

random.seed(version=2)

for i in range(10):
    j = random.randint(0, len(alphas) - 1)
    k = random.randint(0, len(machs) - 1)
    msg = "alpha="
    msg += str(alphas[j])
    msg += ", mach="
    msg += str(machs[k])
    msg += ", cya="
    msg += str(cya[j][k])
    print(msg)
