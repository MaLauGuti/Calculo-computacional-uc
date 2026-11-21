import math
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from gauss_simple import eliminacion_gaussiana_simple
from gauss_banda import eliminacion_gaussiana_banda


def construir_sistema_80(n: int = 80):
    """Construye A (80x80) y b (b_i = pi) según las reglas del Problema 3."""
    A = np.zeros((n, n), dtype=float)
    for idx in range(n):
        i = idx + 1  # Índice matemático 1-based (1..80)
        A[idx, idx] = 2.0 * i
        if idx + 2 < n:
            A[idx, idx + 2] = 0.5 * i
        if idx - 2 >= 0:
            A[idx, idx - 2] = 0.5 * i
        if idx + 4 < n:
            A[idx, idx + 4] = 0.25 * i
        if idx - 4 >= 0:
            A[idx, idx - 4] = 0.25 * i

    b = np.full(n, math.pi, dtype=float)
    return A, b


def generar_figuras(A: np.ndarray, x_sol: np.ndarray, ops_simple: tuple, ops_banda: tuple):
    os.makedirs("figuras", exist_ok=True)

    # Figura 1: Patrón de dispersión (sparsity) de A y vector solución x
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    ax1.spy(A, markersize=3.5, color="#1f77b4")
    ax1.set_title("Estructura de Dispersión de $A$ ($80 \\times 80$)\n(Semiancho de banda $p=q=4$)", fontsize=11)
    ax1.set_xlabel("Columna $j$")
    ax1.set_ylabel("Fila $i$")

    indices = np.arange(1, len(x_sol) + 1)
    ax2.plot(indices, x_sol, "o-", color="#d62728", markersize=3.5, linewidth=1.2)
    ax2.set_title("Solución del Sistema $Ax = b$ ($n=80$, $b_i=\\pi$)", fontsize=11)
    ax2.set_xlabel("Índice $i$")
    ax2.set_ylabel("$x_i$")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("figuras/p3_estructura_y_solucion.png", dpi=300)
    plt.close()

    # Figura 2: Comparación de operaciones aritméticas (Problema 3 vs Problema 4)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    categorias = ["Multiplicaciones / Divisiones", "Sumas / Restas", "Total FLOPs"]
    vals_simple = [ops_simple[0], ops_simple[1], sum(ops_simple)]
    vals_banda = [ops_banda[0], ops_banda[1], sum(ops_banda)]

    x_pos = np.arange(len(categorias))
    ancho = 0.35

    rects1 = ax.bar(x_pos - ancho / 2, vals_simple, ancho, label="Gauss Simple (Problema 3)", color="#4c72b0")
    rects2 = ax.bar(x_pos + ancho / 2, vals_banda, ancho, label="Gauss Banda Optimizado (Problema 4)", color="#55a868")

    ax.set_yscale("log")
    ax.set_ylabel("Número de operaciones (escala logarítmica)")
    ax.set_title("Comparación de Complejidad Computacional ($n = 80$)", fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categorias)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend()

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:,}", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:,}", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

    plt.tight_layout()
    plt.savefig("figuras/p4_comparacion_operaciones.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    os.makedirs("resultados", exist_ok=True)
    n = 80
    A, b = construir_sistema_80(n)

    # Referencia NumPy
    x_ref = np.linalg.solve(A, b)

    # Problema 3: Gauss Simple
    t0 = time.perf_counter()
    x_simple, md_simple, as_simple = eliminacion_gaussiana_simple(A, b)
    t_simple = (time.perf_counter() - t0) * 1000.0  # ms

    # Problema 4: Gauss Banda Optimizado
    t0 = time.perf_counter()
    x_banda, md_banda, as_banda = eliminacion_gaussiana_banda(A, b, p=4, q=4)
    t_banda = (time.perf_counter() - t0) * 1000.0  # ms

    # Fórmulas teóricas exactas de Gauss simple denso:
    # Mul/Div = n(n-1)(2n+5)/6 + n(n+1)/2 = (2n^3 + 3n^2 - 5n)/6 + (n^2+n)/2 = n^3/3 + n^2 - n/3
    teor_md_simple = (n**3 + 3 * n**2 - n) // 3
    teor_as_simple = (2 * n**3 + 3 * n**2 - 5 * n) // 6

    err_simple = np.linalg.norm(A @ x_simple - b, np.inf)
    err_banda = np.linalg.norm(A @ x_banda - b, np.inf)
    diff_sols = np.linalg.norm(x_simple - x_banda, np.inf)

    total_simple = md_simple + as_simple
    total_banda = md_banda + as_banda
    reduccion_pct = (1.0 - total_banda / total_simple) * 100.0

    lineas = [
        "=================================================================",
        "PROBLEMAS 3 Y 4 - SISTEMA BANDA 80x80 (b_i = pi)",
        "=================================================================",
        f"Entradas no nulas de A : {np.count_nonzero(A)} de {n*n} ({np.count_nonzero(A)/(n*n)*100:.2f}%)",
        "",
        "--- PROBLEMA 3: ELIMINACIÓN GAUSSIANA SIMPLE ---",
        f"Multiplicaciones / Divisiones : {md_simple:,} (Teórico: {teor_md_simple:,})",
        f"Sumas / Restas                : {as_simple:,} (Teórico: {teor_as_simple:,})",
        f"Total operaciones             : {total_simple:,}",
        f"Tiempo de ejecución           : {t_simple:.4f} ms",
        f"Residuo ||Ax - b||_inf        : {err_simple:.4e}",
        "",
        "--- PROBLEMA 4: ELIMINACIÓN GAUSSIANA OPTIMIZADA (BANDA p=q=4) ---",
        f"Multiplicaciones / Divisiones : {md_banda:,}",
        f"Sumas / Restas                : {as_banda:,}",
        f"Total operaciones             : {total_banda:,}",
        f"Tiempo de ejecución           : {t_banda:.4f} ms",
        f"Residuo ||Ax - b||_inf        : {err_banda:.4e}",
        "",
        "--- IMPACTO DE LA OPTIMIZACIÓN ---",
        f"Diferencia máxima ||x_simple - x_banda||_inf : {diff_sols:.4e}",
        f"Reducción total de operaciones               : {reduccion_pct:.2f}% ({total_simple // total_banda}x menos operaciones)",
        "",
        "--- VECTOR SOLUCIÓN COMPLETO x (i = 1..80) ---"
    ]

    for i in range(n):
        lineas.append(f"  x_{i+1:02d} = {x_banda[i]:.12f}")

    texto = "\n".join(lineas)
    print(texto)
    with open("resultados/p3_p4_resultados_sistema80.txt", "w", encoding="utf-8") as f:
        f.write(texto)

    generar_figuras(A, x_banda, (md_simple, as_simple), (md_banda, as_banda))
    print("\nFiguras guardadas en figuras/p3_estructura_y_solucion.png y figuras/p4_comparacion_operaciones.png")