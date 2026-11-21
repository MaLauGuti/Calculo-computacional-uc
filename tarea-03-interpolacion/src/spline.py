import numpy as np


def spline_cubico_natural(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Construye el interpolante de Spline Cúbico Natural S(x) siguiendo
    exactamente el Algoritmo 3.4 de Burden & Faires (Numerical Analysis).

    Para cada subintervalo [x_j, x_{j+1}], con j = 0, 1, ..., n - 1:
        S_j(x) = a_j + b_j*(x - x_j) + c_j*(x - x_j)^2 + d_j*(x - x_j)^3

    Parámetros:
        x: arreglo 1D estrictamente creciente con los nodos x_0, x_1, ..., x_n
        y: arreglo 1D con los valores a_i = f(x_i) para i = 0, 1, ..., n

    Retorna:
        a, b, c, d: arreglos 1D de longitud n con los coeficientes de cada tramo
    """
    n = len(x) - 1  # n subintervalos para n + 1 nodos
    a = np.array(y, dtype=float)
    b = np.zeros(n, dtype=float)
    c = np.zeros(n + 1, dtype=float)
    d = np.zeros(n, dtype=float)

    # Paso 1: h_i = x_{i+1} - x_i para i = 0, 1, ..., n - 1
    h = np.zeros(n, dtype=float)
    for i in range(n):
        h[i] = x[i + 1] - x[i]

    # Paso 2: alpha_i para i = 1, 2, ..., n - 1
    alpha = np.zeros(n + 1, dtype=float)
    for i in range(1, n):
        alpha[i] = (3.0 / h[i]) * (a[i + 1] - a[i]) - (3.0 / h[i - 1]) * (a[i] - a[i - 1])

    # Paso 3: Inicialización frontera natural (l_0 = 1, mu_0 = 0, z_0 = 0)
    l = np.zeros(n + 1, dtype=float)
    mu = np.zeros(n + 1, dtype=float)
    z = np.zeros(n + 1, dtype=float)

    l[0] = 1.0
    mu[0] = 0.0
    z[0] = 0.0

    # Paso 4: Eliminación hacia adelante (Algoritmo 6.7 tridiagonal)
    for i in range(1, n):
        l[i] = 2.0 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

    # Paso 5: Condición frontera natural derecha (l_n = 1, z_n = 0, c_n = 0)
    l[n] = 1.0
    z[n] = 0.0
    c[n] = 0.0

    # Paso 6: Sustitución hacia atrás y cálculo de b_j, c_j, d_j
    for j in range(n - 1, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (a[j + 1] - a[j]) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0
        d[j] = (c[j + 1] - c[j]) / (3.0 * h[j])

    # Retornamos a[0:n], b[0:n], c[0:n], d[0:n] correspondientes a j = 0, ..., n - 1
    return a[:n].copy(), b, c[:n].copy(), d


def evaluar_spline_cubico(
    x_nodos: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    x_eval: np.ndarray | float,
) -> np.ndarray | float:
    """
    Evalúa el spline cúbico natural S(x) en un punto escalar o un arreglo de puntos.
    Utiliza forma anidada (Horner) por tramos:
        S_j(x) = a_j + dx * (b_j + dx * (c_j + dx * d_j)), donde dx = x - x_j
    """
    n = len(a)
    es_escalar = np.isscalar(x_eval)
    x_arr = np.atleast_1d(np.asarray(x_eval, dtype=float))
    resultado = np.zeros_like(x_arr, dtype=float)

    for idx, val in enumerate(x_arr):
        # Localizar el subintervalo j tal que x_j <= val <= x_{j+1}
        if val <= x_nodos[0]:
            j = 0
        elif val >= x_nodos[-1]:
            j = n - 1
        else:
            # np.searchsorted(..., side="right") - 1 ubica el índice j correcto
            j = int(np.searchsorted(x_nodos, val, side="right") - 1)
            j = max(0, min(j, n - 1))

        dx = val - x_nodos[j]
        resultado[idx] = a[j] + dx * (b[j] + dx * (c[j] + dx * d[j]))

    if es_escalar:
        return float(resultado[0])
    return resultado