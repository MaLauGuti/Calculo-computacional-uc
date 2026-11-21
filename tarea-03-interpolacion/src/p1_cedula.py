import os
import numpy as np
import matplotlib.pyplot as plt
from newton import diferencias_divididas_newton, evaluar_polinomio_newton


def main():
    # TODO: Sustituye esta lista con los 8 dígitos de tu cédula [d0, d1, d2, d3, d4, d5, d6, d7]
    cedula_digitos = [2, 3, 4, 2, 4, 5, 9, 7]

    x_nodos = np.arange(0, 8, dtype=float)
    y_nodos = np.array(cedula_digitos, dtype=float)

    # 1. Calcular coeficientes c_{0,i} y tabla completa
    coeficientes, tabla = diferencias_divididas_newton(x_nodos, y_nodos)

    # 2. Guardar resultados en texto para el informe
    os.makedirs("tarea-03-interpolacion/resultados", exist_ok=True)
    os.makedirs("tarea-03-interpolacion/figuras", exist_ok=True)

    ruta_txt = "tarea-03-interpolacion/resultados/p1_coeficientes.txt"
    with open(ruta_txt, "w", encoding="utf-8") as f:
        f.write("=== PROBLEMA 1: DIFERENCIAS DIVIDIDAS DE NEWTON (CÉDULA) ===\n\n")
        f.write("Puntos interpolados (x_i, d_i):\n")
        for xi, yi in zip(x_nodos, y_nodos):
            f.write(f"  x = {int(xi)}  ->  y = {int(yi)}\n")

        f.write("\nCoeficientes del polinomio P(x) (c_{0,i}):\n")
        for i, c in enumerate(coeficientes):
            f.write(f"  c_0{i} = {c:.10f}\n")

    print("Coeficientes obtenidos (c_{0,i}):")
    for i, c in enumerate(coeficientes):
        print(f"  c_0{i} = {c:.10f}")

    # 3. Evaluar en 200 puntos igualmente espaciados en [-2, 9]
    x_eval = np.linspace(-2, 9, 200)
    y_eval = evaluar_polinomio_newton(coeficientes, x_nodos, x_eval)

    # 4. Graficar
    plt.figure(figsize=(10, 6))
    plt.plot(x_eval, y_eval, label="Polinomio interpolante $P_7(x)$", color="#1f77b4", linewidth=2)
    plt.scatter(x_nodos, y_nodos, color="#d62728", zorder=5, s=60, label="Nodos $(i, d_i)$")

    plt.title("Interpolación de Newton — Dígitos de Cédula en $[-2, 9]$", fontsize=13)
    plt.xlabel("$x$", fontsize=11)
    plt.ylabel("$P(x)$", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    ruta_fig = "tarea-03-interpolacion/figuras/p1_newton_cedula.png"
    plt.savefig(ruta_fig, dpi=300)
    plt.show()
    print(f"\nArchivos guardados en:\n - {ruta_txt}\n - {ruta_fig}")


if __name__ == "__main__":
    main()