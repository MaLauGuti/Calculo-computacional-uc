import os
import numpy as np
import matplotlib.pyplot as plt
from newton import diferencias_divididas_newton, evaluar_polinomio_newton


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

    ruta_txt = "tarea-03-interpolacion/resultados/p2_coeficientes.txt"
    x_fino = np.linspace(-limite, limite, 500)

    funciones = [
        ("f(x)", f, "p2_newton_fx.png", r"$f(x) = \frac{1}{1 + (5x/12)^2}$"),
        ("g(x)", g, "p2_newton_gx.png", r"$g(x) = \frac{1 + \cos(\pi x / 12)}{2}$"),
    ]

    with open(ruta_txt, "w", encoding="utf-8") as archivo:
        archivo.write("=== PROBLEMA 2: APROXIMACIÓN POLINÓMICA DE NEWTON EN [-12, 12] ===\n")
        archivo.write(f"Último dígito d = {d}  ->  Intervalo = [-{limite}, {limite}]\n\n")

        for nombre_fn, fn, nombre_img, titulo_latex in funciones:
            archivo.write("=" * 65 + "\n")
            archivo.write(f"FUNCIÓN: {nombre_fn}\n")
            archivo.write("=" * 65 + "\n\n")

            fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=False)
            y_exacta = fn(x_fino, d)

            for idx, n in enumerate(valores_n):
                # n + 1 puntos igualmente espaciados en [-12, 12]
                x_nodos = np.linspace(-limite, limite, n + 1)
                y_nodos = fn(x_nodos, d)

                coeficientes, _ = diferencias_divididas_newton(x_nodos, y_nodos)
                y_aprox = evaluar_polinomio_newton(coeficientes, x_nodos, x_fino)

                # Guardar coeficientes c_{0,i} en el archivo de texto
                archivo.write(f"--- Grado n = {n} ({n + 1} nodos) ---\n")
                for i, c in enumerate(coeficientes):
                    archivo.write(f"  c_0{i:<2} = {c: .10f}\n")
                archivo.write("\n")

                # Subgráfica para cada n
                ax = axes[idx]
                ax.plot(x_fino, y_exacta, "k--", linewidth=1.5, label=f"{nombre_fn} exacta")
                ax.plot(x_fino, y_aprox, color="#1f77b4", linewidth=2, label=f"$P_{{{n}}}(x)$")
                ax.scatter(x_nodos, y_nodos, color="#d62728", zorder=5, s=35, label="Nodos")
                ax.set_title(f"$n = {n}$ ({n + 1} puntos)", fontsize=11)
                ax.set_xlabel("$x$")
                if idx == 0:
                    ax.set_ylabel("$y$")
                ax.grid(True, linestyle="--", alpha=0.5)
                ax.legend(fontsize=8.5)

            fig.suptitle(f"Interpolación de Newton — {titulo_latex} en $[-12, 12]$", fontsize=13)
            fig.tight_layout()
            ruta_fig = f"tarea-03-interpolacion/figuras/{nombre_img}"
            fig.savefig(ruta_fig, dpi=300)
            plt.show()
            print(f"Gráfica guardada: {ruta_fig}")

    print(f"Coeficientes guardados en: {ruta_txt}")


if __name__ == "__main__":
    main()