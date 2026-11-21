import math
import os
import numpy as np


def trunc3(val: float) -> float:
    """Trunca un número real a 3 cifras significativas."""
    if val == 0.0:
        return 0.0
    signo = -1.0 if val < 0 else 1.0
    v = abs(val)
    exp = math.floor(math.log10(v))
    factor = 10.0 ** (2 - exp)
    truncado = math.floor(v * factor + 1e-12) / factor
    return signo * truncado


def fl_add(a: float, b: float) -> float:
    return trunc3(trunc3(a) + trunc3(b))


def fl_sub(a: float, b: float) -> float:
    return trunc3(trunc3(a) - trunc3(b))


def fl_mul(a: float, b: float) -> float:
    return trunc3(trunc3(a) * trunc3(b))


def fl_div(a: float, b: float) -> float:
    return trunc3(trunc3(a) / trunc3(b))


def imprimir_matriz(M: np.ndarray, titulo: str, out_lines: list):
    out_lines.append(f"\n--- {titulo} ---")
    for fila in M:
        linea = "  ".join(f"{val:10.4g}" for val in fila)
        out_lines.append(linea)


def construir_sistema_truncado():
    A_exact = np.array([
        [math.pi,      -math.e,       math.sqrt(2), -math.sqrt(3)],
        [math.pi**2,    math.e,      -math.e**2,     3.0 / 7.0],
        [math.sqrt(5), -math.sqrt(6), 1.0,          -math.sqrt(2)],
        [math.pi**3,    math.e**2,   -math.sqrt(7),  1.0 / 9.0]
    ], dtype=float)

    # Últimas 4 cifras de V-23.424.597 -> 4, 5, 9, 7
    b_exact = np.array([4.0, 5.0, 9.0, 7.0], dtype=float)

    A_tr = np.vectorize(trunc3)(A_exact)
    b_tr = np.vectorize(trunc3)(b_exact)
    return A_exact, b_exact, A_tr, b_tr


def gauss_simple_trunc3(A_in: np.ndarray, b_in: np.ndarray, out_lines: list):
    n = len(b_in)
    M = np.column_stack((A_in.copy(), b_in.copy()))
    imprimir_matriz(M, "ELIMINACIÓN GAUSSIANA SIMPLE - Matriz Aumentada Inicial (truncada a 3 cifras)", out_lines)

    for k in range(n - 1):
        for i in range(k + 1, n):
            m = fl_div(M[i, k], M[k, k])
            out_lines.append(f"\nPaso k={k+1}, fila i={i+1}: m_{{{i+1},{k+1}}} = fl({M[i, k]:.4g} / {M[k, k]:.4g}) = {m:.4g}")
            M[i, k] = 0.0
            for j in range(k + 1, n + 1):
                prod = fl_mul(m, M[k, j])
                M[i, j] = fl_sub(M[i, j], prod)
        imprimir_matriz(M, f"Matriz tras etapa k={k+1}", out_lines)

    x = np.zeros(n)
    out_lines.append("\nSustitución regresiva (Simple):")
    for i in range(n - 1, -1, -1):
        suma = M[i, n]
        for j in range(i + 1, n):
            prod = fl_mul(M[i, j], x[j])
            suma = fl_sub(suma, prod)
        x[i] = fl_div(suma, M[i, i])
        out_lines.append(f"  x_{i+1} = fl({suma:.4g} / {M[i, i]:.4g}) = {x[i]:.4g}")

    return x


def gauss_pivoteo_completo_trunc3(A_in: np.ndarray, b_in: np.ndarray, out_lines: list):
    n = len(b_in)
    M = np.column_stack((A_in.copy(), b_in.copy()))
    orden_vars = list(range(n))
    imprimir_matriz(M, "ELIMINACIÓN GAUSSIANA CON PIVOTEO MÁXIMO (COMPLETO) - Matriz Inicial", out_lines)

    for k in range(n - 1):
        sub = np.abs(M[k:, k:n])
        idx_max = np.unravel_index(np.argmax(sub), sub.shape)
        p_fila = k + idx_max[0]
        p_col = k + idx_max[1]

        if p_fila != k:
            M[[k, p_fila]] = M[[p_fila, k]]
        if p_col != k:
            M[:, [k, p_col]] = M[:, [p_col, k]]
            orden_vars[k], orden_vars[p_col] = orden_vars[p_col], orden_vars[k]

        out_lines.append(
            f"\nEtapa k={k+1}: Pivote máximo absoluto = {M[k, k]:.4g} "
            f"(fila {p_fila+1}, col {p_col+1}) | Orden variables: {[v+1 for v in orden_vars]}"
        )

        for i in range(k + 1, n):
            m = fl_div(M[i, k], M[k, k])
            out_lines.append(f"  m_{{{i+1},{k+1}}} = fl({M[i, k]:.4g} / {M[k, k]:.4g}) = {m:.4g}")
            M[i, k] = 0.0
            for j in range(k + 1, n + 1):
                prod = fl_mul(m, M[k, j])
                M[i, j] = fl_sub(M[i, j], prod)
        imprimir_matriz(M, f"Matriz tras etapa k={k+1}", out_lines)

    x_perm = np.zeros(n)
    out_lines.append("\nSustitución regresiva (Pivoteo Completo):")
    for i in range(n - 1, -1, -1):
        suma = M[i, n]
        for j in range(i + 1, n):
            prod = fl_mul(M[i, j], x_perm[j])
            suma = fl_sub(suma, prod)
        x_perm[i] = fl_div(suma, M[i, i])
        out_lines.append(f"  x_perm_{i+1} (x_{orden_vars[i]+1}) = fl({suma:.4g} / {M[i, i]:.4g}) = {x_perm[i]:.4g}")

    x = np.zeros(n)
    for pos, var_idx in enumerate(orden_vars):
        x[var_idx] = x_perm[pos]

    return x


if __name__ == "__main__":
    os.makedirs("resultados", exist_ok=True)
    out = []

    A_ex, b_ex, A_tr, b_tr = construir_sistema_truncado()
    x_exact = np.linalg.solve(A_ex, b_ex)

    out.append("SOLUCIÓN EXACTA (doble precisión IEEE 754):")
    for i, val in enumerate(x_exact, 1):
        out.append(f"  x_{i} = {val:.10f}")

    x_simp = gauss_simple_trunc3(A_tr, b_tr, out)
    x_comp = gauss_pivoteo_completo_trunc3(A_tr, b_tr, out)

    out.append("\n================ RESUMEN COMPARATIVO (PROBLEMA 1) ================")
    out.append(f"Exacta           : {x_exact}")
    out.append(f"Gauss Simple     : {x_simp} | Error rel L_inf = {np.linalg.norm(x_exact - x_simp, np.inf) / np.linalg.norm(x_exact, np.inf):.4e}")
    out.append(f"Pivoteo Completo : {x_comp} | Error rel L_inf = {np.linalg.norm(x_exact - x_comp, np.inf) / np.linalg.norm(x_exact, np.inf):.4e}")

    texto = "\n".join(out)
    print(texto)
    with open("resultados/p1_pasos_truncamiento.txt", "w", encoding="utf-8") as f:
        f.write(texto)