import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Cargar los datos procesados
df_resultados = pd.read_csv("resultados_abs_err.csv")
# Calcular el máximo del valor absoluto del error para cada configuración
df_max_error = df_resultados.groupby(["DeltaR", "Ángulo", "Posición"]) ["MSE"].max().reset_index()

# Ordenar por DeltaR para mejor visualización
df_max_error = df_max_error.sort_values(by="DeltaR")

# Asignar distintos marcadores según la posición
posiciones = df_max_error["Posición"].unique()
markers = ["o", "s", "^", "D", "v", "*"]  # Diferentes formas
marker_dict = {pos: markers[i % len(markers)] for i, pos in enumerate(posiciones)}

# Asignar colores según el ángulo
angulos = df_max_error["Ángulo"].unique()
colores = plt.cm.get_cmap("rainbow", len(angulos))
color_dict = {angulo: colores(i) for i, angulo in enumerate(angulos)}

# Crear gráfico scatter
plt.figure(figsize=(10, 6))
for angulo in angulos:
    subset_angulo = df_max_error[df_max_error["Ángulo"] == angulo]
    for pos in posiciones:
        subset = subset_angulo[subset_angulo["Posición"] == pos]
        plt.scatter(subset["DeltaR"], subset["MSE"], color=color_dict[angulo], s=100,
                    edgecolors='k', alpha=0.75, marker=marker_dict[pos], label=f"Ángulo {angulo}, Pos {pos}")

# Agregar leyenda con colores y formas
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker=marker_dict[pos], color='w', label=f"Posición {pos}",
                          markerfacecolor='gray', markersize=10, linestyle='None') for pos in posiciones]
legend_colors = [Line2D([0], [0], marker='o', color='w', label=f"Ángulo {angulo}",
                         markerfacecolor=color_dict[angulo], markersize=10, linestyle='None') for angulo in angulos]

plt.legend(handles=legend_elements + legend_colors, title="Configuraciones", loc="upper right")

plt.xlabel("DeltaR (cm)")
plt.ylabel("Máximo |Error|")
plt.title("Máximo del valor absoluto del error - Todos los anchos de banda")
plt.grid(True)
plt.show()