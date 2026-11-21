import numpy as np


def eliminacion_gaussiana_banda(A_in: np.ndarray, b_in: np.ndarray, p: int = 4, q: int = 4):
    """
    Eliminación Gaussiana optimizada para matrices banda (Problema 4).
    Explota el semiancho de banda inferior p=4 y superior q=4, y omite
    entradas nulas dentro de la banda (diagonales impares).
    Retorna: (x, ops_mul_div, ops_add_sub)
    """
    n = len(b_in)
    A = A_in.astype(float).copy()
    b = b_in.astype(float).copy()

    mul_div = 0
    add_sub = 0

    # Fase de eliminación progresiva acotada a la banda [p, q]
    for k in range(n - 1):
        pivote = A[k, k]
        if pivote == 0.0:
            raise ZeroDivisionError(f"Pivote nulo en etapa k={k}")

        i_max = min(n - 1, k + p)
        j_max = min(n - 1, k + q)

        for i in range(k + 1, i_max + 1):
            # Si el elemento debajo del pivote ya es cero, no hay nada que eliminar
            if A[i, k] == 0.0:
                continue

            factor = A[i, k] / pivote
            mul_div += 1
            A[i, k] = 0.0

            for j in range(k + 1, j_max + 1):
                # Evitar multiplicar por cero si A[k, j] == 0 (ej. diagonales impares)
                if A[k, j] != 0.0:
                    A[i, j] -= factor * A[k, j]
                    mul_div += 1
                    add_sub += 1

            b[i] -= factor * b[k]
            mul_div += 1
            add_sub += 1

    # Fase de sustitución regresiva acotada al semiancho superior q
    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        suma = b[i]
        j_max = min(n - 1, i + q)
        for j in range(i + 1, j_max + 1):
            if A[i, j] != 0.0:
                suma -= A[i, j] * x[j]
                mul_div += 1
                add_sub += 1
        x[i] = suma / A[i, i]
        mul_div += 1

    return x, mul_div, add_sub