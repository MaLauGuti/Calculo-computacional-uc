import os
import numpy as np
import matplotlib.pyplot as plt
from spline import spline_cubico_natural, evaluar_spline_cubico


def f(x: np.ndarray, d: int = 7) -> np.ndarray:
    return 1.0 / (1.0 + ((5.0 * x) / (d + 5)) ** 2)


def g(x: np.ndarray, d: int = 7) -> np.ndarray:
    return (1.0 + np.cos((np.pi * x) / (d + 5))) / 2.0


def main():
    d = 7  # Último dígito de la cédula 23424597
    limite = d + 5  # Intervalo [-12, 12]
    valores_n = [4, 7, 10]

    os.makedirs("tarea-03-interpolacion/resultados", exist_ok=True)
    os.makedirs("tarea-03-interpolacion/figuras", exist_ok=True)

    ruta_txt = "tarea-03-interpolacion/resultados/p3_coeficientes.txt"
    x_fino = np.linspace(-limite, limite, 500)

    funciones = [
        ("f(x)", f, "p3_spline_fx.png", r"$f(x) = \frac{1}{1 + (5x/12)^2}$"),
        ("g(x)", g, "p3_spline_gx.png", r"$g(x) = \frac{1 + \cos(\pi x / 12)}{2}$"),
    ]

    with open(ruta_txt, "w", encoding="utf-8") as archivo:
        archivo.write("=== PROBLEMA 3: SPLINE CÚBICO NATURAL EN [-12, 12] ===\n")
        archivo.write(f"Último dígito d = {d}  ->  Intervalo = [-{limite}, {limite}]\n")
        archivo.write("Forma por tramos: S_j(x) = a_j + b_j(x - x_j) + c_j(x - x_j)^2 + d_j(x - x_j)^3\n\n")

        for nombre_fn, fn, nombre_img, titulo_latex in funciones:
            archivo.write("=" * 82 + "\n")
            archivo.write(f"FUNCIÓN: {nombre_fn}\n")
            archivo.write("=" * 82 + "\n\n")

            fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=False)
            y_exacta = fn(x_fino, d)

            for idx, n in enumerate(valores_n):
                # n + 1 nodos igualmente espaciados -> n subintervalos
                x_nodos = np.linspace(-limite, limite, n + 1)
                y_nodos = fn(x_nodos, d)

                a, b, c, d_coef = spline_cubico_natural(x_nodos, y_nodos)
                y_aprox = evaluar_spline_cubico(x_nodos, a, b, c, d_coef, x_fino)

                # Guardar tabla de coeficientes por tramo j = 0, ..., n - 1
                archivo.write(f"--- n = {n} ({n + 1} nodos, {n} tramos) ---\n")
                archivo.write(
                    f"{'j':>3} | {'x_j':>10} | {'a_j':>14} | {'b_j':>14} | {'c_j':>14} | {'d_j':>14}\n"
                )
                archivo.write("-" * 82 + "\n")
                for j in range(n):
                    archivo.write(
                        f"{j:>3} | {x_nodos[j]:>10.4f} | {a[j]:>14.8f} | "
                        f"{b[j]:>14.8f} | {c[j]:>14.8f} | {d_coef[j]:>14.8f}\n"
                    )
                archivo.write("\n")

                # Subgráfica para cada n
                ax = axes[idx]
                ax.plot(x_fino, y_exacta, "k--", linewidth=1.5, label=f"{nombre_fn} exacta")
                ax.plot(x_fino, y_aprox, color="#2ca02c", linewidth=2, label=f"$S_{{{n}}}(x)$ (Spline)")
                ax.scatter(x_nodos, y_nodos, color="#d62728", zorder=5, s=35, label="Nodos")
                ax.set_title(f"$n = {n}$ ({n + 1} puntos)", fontsize=11)
                ax.set_xlabel("$x$")
                if idx == 0:
                    ax.set_ylabel("$y$")
                ax.grid(True, linestyle="--", alpha=0.5)
                ax.legend(fontsize=8.5)

            fig.suptitle(f"Spline Cúbico Natural — {titulo_latex} en $[-12, 12]$", fontsize=13)
            fig.tight_layout()
            ruta_fig = f"tarea-03-interpolacion/figuras/{nombre_img}"
            fig.savefig(ruta_fig, dpi=300)
            plt.show()
            print(f"Gráfica guardada: {ruta_fig}")

    print(f"Coeficientes guardados en: {ruta_txt}")


if __name__ == "__main__":
    main()