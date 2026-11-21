import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def main():
    ruta_img = "tarea-03-interpolacion/assets/tipografia_old_english.jpg"
    if not os.path.exists(ruta_img):
        # Busca cualquier imagen en assets por si tiene otro nombre
        carpeta = "tarea-03-interpolacion/assets"
        archivos = [f for f in os.listdir(carpeta) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if not archivos:
            raise FileNotFoundError("No se encontró ninguna imagen en tarea-03-interpolacion/assets/")
        ruta_img = os.path.join(carpeta, archivos[0])

    img = mpimg.imread(ruta_img)
    alto, ancho = img.shape[:2]

    # La letra L está en la fila 2 (de 3) y columna 3 (de 9)
    y_min_px, y_max_px = int(alto * 0.34), int(alto * 0.66)
    x_min_px, x_max_px = int(ancho * (2.0 / 9.0)), int(ancho * (3.15 / 9.0))
    recorte = img[y_min_px:y_max_px, x_min_px:x_max_px]

    fig, ax = plt.subplots(figsize=(7, 7))
    # Proyectamos la letra en un sistema de coordenadas limpio [0, 10] x [0, 10]
    ax.imshow(recorte, extent=[0, 10, 0, 10], aspect="equal")
    ax.set_xticks(np.arange(0, 10.5, 1))
    ax.set_yticks(np.arange(0, 10.5, 1))
    ax.grid(True, color="royalblue", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.set_title(
        "Haz clic siguiendo el trazo de la 'L' (mínimo 16 puntos)\n"
        "Clic izq: agregar | Clic der / Retroceso: borrar último | Enter: terminar",
        fontsize=10.5,
    )
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

    puntos = []
    scatter = ax.scatter([], [], color="crimson", s=45, zorder=5)
    linea, = ax.plot([], [], color="crimson", linestyle=":", linewidth=1.2, zorder=4)
    etiquetas = []

    def actualizar():
        if puntos:
            arr = np.array(puntos)
            scatter.set_offsets(arr)
            linea.set_data(arr[:, 0], arr[:, 1])
        else:
            scatter.set_offsets(np.empty((0, 2)))
            linea.set_data([], [])

        for txt in etiquetas:
            txt.remove()
        etiquetas.clear()
        for idx, (px, py) in enumerate(puntos):
            etiquetas.append(
                ax.text(px + 0.12, py + 0.12, str(idx), color="darkred", fontsize=8, fontweight="bold")
            )
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax:
            return
        if event.button == 1:
            puntos.append((round(event.xdata, 3), round(event.ydata, 3)))
            actualizar()
        elif event.button == 3 and puntos:
            puntos.pop()
            actualizar()

    def on_key(event):
        if event.key in ("backspace", "delete") and puntos:
            puntos.pop()
            actualizar()
        elif event.key == "enter":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.tight_layout()
    plt.show()

    os.makedirs("tarea-03-interpolacion/resultados", exist_ok=True)
    ruta_salida = "tarea-03-interpolacion/resultados/p4_puntos_capturados.txt"
    with open(ruta_salida, "w", encoding="utf-8") as f:
        f.write(f"# Total de puntos capturados: {len(puntos)}\n")
        f.write("PUNTOS_L = [\n")
        for px, py in puntos:
            f.write(f"    ({px:.3f}, {py:.3f}),\n")
        f.write("]\n")

    print(f"\n¡Listo! Se capturaron {len(puntos)} puntos y se guardaron en: {ruta_salida}")


if __name__ == "__main__":
    main()