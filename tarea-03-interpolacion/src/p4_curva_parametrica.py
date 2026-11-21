import os
import ast
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from spline import spline_cubico_natural, evaluar_spline_cubico


def cargar_puntos(ruta_txt: str) -> np.ndarray:
    with open(ruta_txt, "r", encoding="utf-8") as f:
        contenido = f.read()
    inicio = contenido.index("[")
    fin = contenido.rindex("]") + 1
    lista = ast.literal_eval(contenido[inicio:fin])
    return np.array(lista, dtype=float)


def parametrizar_longitud_arco(puntos: np.ndarray) -> np.ndarray:
    """
    Calcula t_0 = 0 y t_i = t_{i-1} + sqrt((x_i - x_{i-1})^2 + (y_i - y_{i-1})^2)
    """
    n = len(puntos) - 1
    t = np.zeros(n + 1, dtype=float)
    for i in range(1, n + 1):
        dx = puntos[i, 0] - puntos[i - 1, 0]
        dy = puntos[i, 1] - puntos[i - 1, 1]
        t[i] = t[i - 1] + np.hypot(dx, dy)
    return t


def escribir_tabla_spline(archivo, nombre: str, t: np.ndarray, a, b, c, d):
    n = len(a)
    archivo.write(f"--- Coeficientes de {nombre}(t) ({n} tramos) ---\n")
    archivo.write(
        f"{'j':>3} | {'t_j':>10} | {'a_j':>14} | {'b_j':>14} | {'c_j':>14} | {'d_j':>14}\n"
    )
    archivo.write("-" * 82 + "\n")
    for j in range(n):
        archivo.write(
            f"{j:>3} | {t[j]:>10.4f} | {a[j]:>14.8f} | "
            f"{b[j]:>14.8f} | {c[j]:>14.8f} | {d[j]:>14.8f}\n"
        )
    archivo.write("\n")


def main():
    ruta_puntos = "tarea-03-interpolacion/resultados/p4_puntos_capturados.txt"
    puntos = cargar_puntos(ruta_puntos)
    if len(puntos) < 16:
        raise ValueError(f"Se requieren al menos 16 puntos; actualmente hay {len(puntos)}.")

    x = puntos[:, 0]
    y = puntos[:, 1]
    t = parametrizar_longitud_arco(puntos)

    # Splines cúbicos naturales independientes: x = S_x(t), y = S_y(t)
    ax_c, bx_c, cx_c, dx_c = spline_cubico_natural(t, x)
    ay_c, by_c, cy_c, dy_c = spline_cubico_natural(t, y)

    t_fino = np.linspace(t[0], t[-1], 1000)
    x_spline = evaluar_spline_cubico(t, ax_c, bx_c, cx_c, dx_c, t_fino)
    y_spline = evaluar_spline_cubico(t, ay_c, by_c, cy_c, dy_c, t_fino)

    # Guardar tabla de nodos (t_i, x_i, y_i) y coeficientes de S_x(t) y S_y(t)
    ruta_txt = "tarea-03-interpolacion/resultados/p4_coeficientes.txt"
    with open(ruta_txt, "w", encoding="utf-8") as archivo:
        archivo.write("=== PROBLEMA 4: RECONSTRUCCIÓN PARAMÉTRICA DE INICIAL 'L' ===\n")
        archivo.write(f"Número de puntos digitalizados: {len(puntos)} (n = {len(puntos) - 1} tramos)\n")
        archivo.write(f"Longitud de arco poligonal total t_n = {t[-1]:.6f}\n\n")

        archivo.write("--- Tabla de Nodos y Parámetro t_i ---\n")
        archivo.write(f"{'i':>3} | {'t_i':>12} | {'x_i':>12} | {'y_i':>12}\n")
        archivo.write("-" * 48 + "\n")
        for i in range(len(puntos)):
            archivo.write(f"{i:>3} | {t[i]:>12.6f} | {x[i]:>12.4f} | {y[i]:>12.4f}\n")
        archivo.write("\n")

        escribir_tabla_spline(archivo, "S_x", t, ax_c, bx_c, cx_c, dx_c)
        escribir_tabla_spline(archivo, "S_y", t, ay_c, by_c, cy_c, dy_c)

    # Gráfica comparativa: izquierda sobre referencia original, derecha curva paramétrica limpia
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))

    carpeta_assets = "tarea-03-interpolacion/assets"
    archivos_img = [f for f in os.listdir(carpeta_assets) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if archivos_img:
        img = mpimg.imread(os.path.join(carpeta_assets, archivos_img[0]))
        alto, ancho = img.shape[:2]
        recorte = img[int(alto * 0.34):int(alto * 0.66), int(ancho * (2.0 / 9.0)):int(ancho * (3.15 / 9.0))]
        axes[0].imshow(recorte, extent=[0, 10, 0, 10], aspect="equal", alpha=0.45)

    axes[0].plot(x, y, "k--", linewidth=1, alpha=0.6, label="Poligonal")
    axes[0].plot(x_spline, y_spline, color="#1f77b4", linewidth=2.2, label="$(S_x(t), S_y(t))$")
    axes[0].scatter(x, y, color="#d62728", s=30, zorder=5, label=f"Nodos ({len(puntos)} pts)")
    axes[0].set_title("Superposición sobre tipografía original", fontsize=11)
    axes[0].set_xlabel("$x$")
    axes[0].set_ylabel("$y$")
    axes[0].set_xlim(0, 10)
    axes[0].set_ylim(0, 10)
    axes[0].set_aspect("equal")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=8.5, loc="lower right")

    axes[1].plot(x_spline, y_spline, color="#1f77b4", linewidth=2.5, label="Curva Spline Cúbica")
    axes[1].scatter(x, y, color="#d62728", s=32, zorder=5, label="Puntos de control")
    axes[1].set_title("Reconstrucción Paramétrica $(S_x(t), S_y(t))$", fontsize=11)
    axes[1].set_xlabel("$x$")
    axes[1].set_ylabel("$y$")
    axes[1].set_xlim(0, 10)
    axes[1].set_ylim(0, 10)
    axes[1].set_aspect("equal")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(fontsize=8.5, loc="lower right")

    fig.suptitle("Interpolación Paramétrica con Splines Cúbicos Naturales — Letra 'L'", fontsize=13)
    fig.tight_layout()
    ruta_fig = "tarea-03-interpolacion/figuras/p4_letra_parametrica.png"
    fig.savefig(ruta_fig, dpi=300)
    plt.show()

    print(f"Coeficientes guardados en: {ruta_txt}")
    print(f"Gráfica guardada en: {ruta_fig}")


if __name__ == "__main__":
    main()