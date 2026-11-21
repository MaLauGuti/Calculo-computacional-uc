import numpy as np


def diferencias_divididas_newton(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula la tabla de diferencias divididas de Newton siguiendo exactamente el algoritmo
    descrito en Kincaid & Cheney (Numerical Analysis, págs. 300-301).

    Parámetros:
        x: arreglo 1D con los nodos x_0, x_1, ..., x_n
        y: arreglo 1D con los valores y_0, y_1, ..., y_n

    Retorna:
        coeficientes: arreglo 1D con los coeficientes c_{0,i} (primera fila de la tabla)
        tabla: matriz 2D triangular superior de diferencias divididas c_{i,j}
    """
    n = len(x) - 1  # Grado máximo del polinomio (para n+1 puntos: 0, 1, ..., n)
    c = np.zeros((n + 1, n + 1), dtype=float)

    # Inicialización: c_{i,0} = y_i para i = 0, 1, ..., n
    for i in range(n + 1):
        c[i, 0] = y[i]

    # Cálculo de diferencias divididas (triangular superior):
    # Para j = 1 hasta n:
    #   Para i = 0 hasta n - j:
    #     c_{i,j} = (c_{i+1, j-1} - c_{i, j-1}) / (x_{i+j} - x_i)
    for j in range(1, n + 1):
        for i in range(n - j + 1):
            c[i, j] = (c[i + 1, j - 1] - c[i, j - 1]) / (x[i + j] - x[i])

    # Los coeficientes del polinomio de Newton son la primera fila: c_{0,0}, c_{0,1}, ..., c_{0,n}
    coeficientes = c[0, :].copy()
    return coeficientes, c


def evaluar_polinomio_newton(
    coeficientes: np.ndarray,
    x_nodos: np.ndarray,
    x_eval: np.ndarray | float,
) -> np.ndarray | float:
    """
    Evalúa el polinomio interpolante de Newton mediante multiplicación anidada
    (algoritmo de evaluación de Kincaid & Cheney, pág. 301).
    """
    n = len(coeficientes) - 1
    if isinstance(x_eval, np.ndarray):
        resultado = np.full_like(x_eval, coeficientes[n], dtype=float)
    else:
        resultado = float(coeficientes[n])

    for i in range(n - 1, -1, -1):
        resultado = resultado * (x_eval - x_nodos[i]) + coeficientes[i]

    return resultado