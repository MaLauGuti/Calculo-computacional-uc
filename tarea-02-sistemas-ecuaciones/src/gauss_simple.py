import numpy as np


def eliminacion_gaussiana_simple(A_in: np.ndarray, b_in: np.ndarray):
    """
    Eliminación Gaussiana Simple sin modificaciones (Problema 3).
    Recorre toda la submatriz (n - k - 1) x (n - k - 1) sin explotar ceros.
    Retorna: (x, ops_mul_div, ops_add_sub)
    """
    n = len(b_in)
    A = A_in.astype(float).copy()
    b = b_in.astype(float).copy()

    mul_div = 0
    add_sub = 0

    # Fase de eliminación progresiva
    for k in range(n - 1):
        pivote = A[k, k]
        if pivote == 0.0:
            raise ZeroDivisionError(f"Pivote nulo en etapa k={k}")

        for i in range(k + 1, n):
            factor = A[i, k] / pivote
            mul_div += 1
            A[i, k] = 0.0

            for j in range(k + 1, n):
                A[i, j] -= factor * A[k, j]
                mul_div += 1
                add_sub += 1

            b[i] -= factor * b[k]
            mul_div += 1
            add_sub += 1

    # Fase de sustitución regresiva
    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        suma = b[i]
        for j in range(i + 1, n):
            suma -= A[i, j] * x[j]
            mul_div += 1
            add_sub += 1
        x[i] = suma / A[i, i]
        mul_div += 1

    return x, mul_div, add_sub