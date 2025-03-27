import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Cargar los datos procesados
df_resultados = pd.read_csv("resultados_abs_err.csv")

# Calcular el valor absoluto del error
df_resultados["Abs_MSE"] = np.abs(df_resultados["MSE"])

'''
# Diferenciado por DeltaR y posición
# Ordenar por DeltaR para mejor visualización
df_resultados = df_resultados.sort_values(by="DeltaR")

# Crear boxplot para el valor absoluto del error
plt.figure(figsize=(12, 6))

# Crear el boxplot
sns.boxplot(data=df_resultados, x="DeltaR", y="Abs_MSE", hue="Posición", palette="Set1")

# Personalizar gráfico
plt.xlabel("DeltaR (cm)")
plt.ylabel("Valor Absoluto del Error")
plt.title("Distribución del Valor Absoluto del Error - Todas las Bandas")
plt.legend(title="Posición", loc="upper left")

# Mostrar el gráfico
plt.grid(True)
plt.show()
'''

'''
# Diferenciado sólo por posición
# Crear el boxplot sin diferenciar por DeltaR
plt.figure(figsize=(10, 6))

# Crear el boxplot
sns.boxplot(data=df_resultados, x="Posición", y="Abs_MSE", palette="Set1")

# Personalizar gráfico
plt.xlabel("Posición")
plt.ylabel("Valor Absoluto del Error")
plt.title("Distribución del Valor Absoluto del Error por Posición - Todas las Bandas")

# Mostrar el gráfico
plt.grid(True)
plt.show()
'''

'''
# Diferenciado sólo por DeltaR
# Crear el boxplot solo diferenciado por DeltaR
plt.figure(figsize=(10, 6))

# Crear el boxplot
sns.boxplot(data=df_resultados, x="DeltaR", y="Abs_MSE", palette="rainbow")

# Personalizar gráfico
plt.xlabel("DeltaR (cm)")
plt.ylabel("Valor Absoluto del Error")
plt.title("Distribución del Valor Absoluto del Error por DeltaR - Todas las Bandas")

# Mostrar el gráfico
plt.grid(True)
plt.show()
'''

# Diferenciado sólo por Ancho de banda
# Crear el boxplot solo diferenciado por DeltaR
plt.figure(figsize=(10, 6))

# Crear el boxplot
sns.boxplot(data=df_resultados, x="Banda", y="Abs_MSE", palette="viridis")

# Personalizar gráfico
plt.xlabel("DeltaR (cm)")
plt.ylabel("Valor Absoluto del Error")
plt.title("Distribución del Valor Absoluto del Error por Ancho de Banda")

# Mostrar el gráfico
plt.grid(True)
plt.show()
