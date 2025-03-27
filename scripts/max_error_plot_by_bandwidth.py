import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Cargar los datos procesados
df_resultados = pd.read_csv("resultados_abs_err.csv")

# Seleccionar el ancho de banda a analizar
banda_seleccionada = "Bandwidth1"  # Cambia esto según el caso
df_filtrado = df_resultados[df_resultados["Banda"] == banda_seleccionada]

# Calcular el máximo del valor absoluto del error para cada configuración
df_max_error = df_filtrado.groupby(["DeltaR", "Ángulo", "Posición"])["MSE"].max().reset_index()

# Ordenar por DeltaR para mejor visualización
df_max_error = df_max_error.sort_values(by="DeltaR")

# Asignar distintos marcadores según la posición
posiciones = df_max_error["Posición"].unique()
markers = ["o", "s", "^", "D", "v", "*"]  # Diferentes formas
marker_dict = {pos: markers[i % len(markers)] for i, pos in enumerate(posiciones)}

# Asignar colores según ángulo
colores = {0: "blue", 30: "red"}

# Crear gráfico scatter
plt.figure(figsize=(10, 6))
for pos in posiciones:
    subset = df_max_error[df_max_error["Posición"] == pos]
    for angulo in colores.keys():
        subset_angulo = subset[subset["Ángulo"] == angulo]
        plt.scatter(subset_angulo["DeltaR"], subset_angulo["MSE"], color=colores[angulo], s=100,
                    edgecolors='k', alpha=0.75, marker=marker_dict[pos], label=f"Posición {pos}, Ángulo {angulo}")

# Agregar leyenda con colores y formas
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker=marker_dict[pos], color='w', label=f"Posición {pos}",
                          markerfacecolor='gray', markersize=10, linestyle='None') for pos in posiciones]
legend_colors = [Line2D([0], [0], marker='o', color='w', label=f"Ángulo {angulo}",
                         markerfacecolor=col, markersize=10, linestyle='None') for angulo, col in colores.items()]

plt.legend(handles=legend_elements + legend_colors, title="Configuraciones", loc="upper right")

plt.xlabel("DeltaR (cm)")
plt.ylabel("Máximo |Error|")
plt.title(f"Máximo del valor absoluto del error - {banda_seleccionada}")
plt.grid(True)
plt.show()