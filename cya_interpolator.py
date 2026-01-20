import math


def solve_slae_seidel(coefficients_matrix, right_hand_column, precision=0.1):
    dim = len(right_hand_column)

    if len(coefficients_matrix) != dim:
        raise ValueError("Wrong slae size")

    for i in range(len(coefficients_matrix)):
        if len(coefficients_matrix[i]) != dim:
            raise ValueError("Wrong slae size")

    # Из лекций Дорошина: Метод Зейделя сходится, если СЛАУ
    # нормальная
    # Нормализуем A^T A X = A^T B
    coefficients_matrix, right_hand_column = normalize_slae(coefficients_matrix, right_hand_column)

    x = right_hand_column

    converge = False
    while not converge:
        x_next = []
        diff = 0
        for i in range(dim):
            x_next.append(0)
            sum1 = sum(coefficients_matrix[i][j] * x_next[j] for j in range(i))
            sum2 = sum(coefficients_matrix[i][j] * x[j] for j in range(i + 1, dim))
            x_next[i] = (right_hand_column[i] - sum1 - sum2) / coefficients_matrix[i][i]
            diff += (x_next[i] - x[i]) ** 2

        x = x_next

        diff = math.sqrt(diff)

        converge = diff < precision

    return x


def transpose_matrix(matrix):
    dim1 = len(matrix)
    dim2 = len(matrix[0])

    result = []

    for i in range(dim2):
        result.append([])
        for j in range(dim1):
            result[i].append(matrix[j][i])

    return result


def normalize_slae(coefficients_matrix, right_hand_column):
    dim = len(coefficients_matrix)
    transposed_coefficients = transpose_matrix(coefficients_matrix)

    new_coefficients = []

    for i in range(dim):
        new_coefficients.append([])
        for j in range(dim):
            new_coefficients[i].append(0)
            for k in range(dim):
                new_coefficients[i][j] += transposed_coefficients[i][k] * coefficients_matrix[k][j]

    new_right_hand_column = []

    for i in range(dim):
        new_right_hand_column.append(0)
        for j in range(dim):
            new_right_hand_column[i] += transposed_coefficients[i][j] * right_hand_column[j]

    return new_coefficients, new_right_hand_column


class ColumnInterpolator:
    def __init__(self, arguments, values):
        # Для линейной функции y=ax+b МНК
        # i = 1,2,3,...,n
        # a = ( n * sum_i^n(x_i*y_i)-sum_i^n(x_i)*sum_i^n(y_i) ) /
        # / ( n * sum_i^n(x_i^2) - (sum_i^n(x_i))^2 )
        # b = sum_i^n(y_i)/n - a*sum_i^n(x_i)/n

        n = len(values)
        sum_x = sum(arguments)
        sum_y = sum(values)
        sum_xy = sum(xi * yi for xi, yi in zip(arguments, values))
        sum_x2 = sum(xi ** 2 for xi in arguments)

        self.a = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        self.b = (sum_y - self.a * sum_x) / n

    def evaluate(self, argument):
        return self.a * argument + self.b


class RowInterpolator:
    def __init__(self, arguments, values):
        self.arguments = arguments

        # Число промежутков между точками
        n = len(values) - 1
        # Коэффициентов интерполяции 4*n

        # Для удобства будем хранить коэффициенты в матрице из n строк и 4 столбцов
        self.interpolation_coefficients = []

        for i in range(n):
            self.interpolation_coefficients.append([])
            for j in range(4):
                self.interpolation_coefficients[i].append(0)

        slae_coefficients = []
        slae_right_hand = []

        # Функции слева и справа совпадают в узловых точках
        for i in range(2 * n):
            slae_right_hand.append(values[(i + 1) // 2])
            x = self.arguments[(i + 1) // 2] - self.arguments[i // 2]
            slae_coefficients.append([])
            for j in range((i // 2) * 4):
                slae_coefficients[i].append(0)

            slae_coefficients[i].extend([1, x, x ** 2, x ** 3])

            for j in range((i // 2 + 1) * 4, 4 * n):
                slae_coefficients[i].append(0)

        # Производные функции слева и справа совпадают в узловых точках
        for i in range(2 * n, 3 * n - 1):
            slae_right_hand.append(0)
            j = i - 2 * n
            x = self.arguments[j + 1] - self.arguments[j]

            slae_coefficients.append([])

            for k in range(j * 4):
                slae_coefficients[i].append(0)

            slae_coefficients[i].extend([0, 1, 2 * x, 3 * x ** 2, 0, 1, 0, 0])

            for k in range((j + 2) * 4, 4 * n):
                slae_coefficients[i].append(0)

        # Вторые производные функции слева и справа совпадают в узловых точках
        for i in range(3 * n - 1, 4 * n - 2):
            slae_right_hand.append(0)
            j = i - (3 * n - 1)
            x = self.arguments[j + 1] - self.arguments[j]

            slae_coefficients.append([])

            for k in range(j * 4):
                slae_coefficients[i].append(0)

            slae_coefficients[i].extend([0, 0, 1, 3 * x, 0, 0, 1, 0])

            for k in range((j + 2) * 4, 4 * n):
                slae_coefficients[i].append(0)

        # Вторые производные на концах равны 0
        slae_right_hand.extend([0, 0])
        x = self.arguments[-1] - self.arguments[-2]
        slae_coefficients.extend([[], []])

        slae_coefficients[-2].extend([0, 0, 1, 0])

        for i in range(4, 4 * n):
            slae_coefficients[-2].append(0)

        for i in range(4 * (n - 1)):
            slae_coefficients[-1].append(0)

        slae_coefficients[-1].extend([0, 0, 1, 3 * x])

        # Решаем СЛАУ, находим коэффициенты интерполяции
        solution = solve_slae_seidel(slae_coefficients, slae_right_hand, 0.001)

        # Распаковываем коэффициенты интерполяции в поле класса
        for i in range(len(solution)):
            self.interpolation_coefficients[i // 4][i % 4] = solution[i]

    def evaluate(self, mach):
        index = self.get_segment_left_border_index(mach)
        a, b, c, d = self.interpolation_coefficients[index]
        x = mach - self.arguments[0]

        return a + b * x + c * x ** 2 + d * x ** 3

    def get_segment_left_border_index(self, mach):
        if not (self.arguments[0] < mach < self.arguments[-1]):
            raise ValueError("Mach number out of range")

        left = 0
        right = len(self.arguments) - 1

        while right - left > 1:
            mid = (left + right) // 2
            if self.arguments[mid] < mach:
                left = mid
            else:
                right = mid

        return left
