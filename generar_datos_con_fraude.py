import os
import csv
import time
from faker import Faker
import random

# Creamos una instancia de Faker
faker = Faker()

# Lista de países, incluyendo algunos de alto riesgo
paises = [
    "Afghanistan", "Iran", "Iraq", "North Korea", "Syria", "Venezuela", "Yemen",
    "Libya", "Somalia", "South Sudan", "Sudan", "Pakistan", "Nigeria",
    "USA", "Canada", "UK", "Germany", "France", "Italy", "Spain", "Australia",
    "Brazil", "India", "China", "Japan"
]

paises_alto_riesgo = [
    "Afghanistan", "Iran", "Iraq", "North Korea", "Syria", "Venezuela", "Yemen",
    "Libya", "Somalia", "South Sudan", "Sudan", "Pakistan", "Nigeria"
]


# Generamos un conjunto de cuentas con perfiles (personal o empresa)
def generar_cuentas(num_cuentas=50000):
    cuentas = []
    for _ in range(num_cuentas):
        cuenta = faker.iban()
        pais = random.choice(paises)
        perfil = random.choices(["personal", "empresa"], [0.7, 0.3])[0]  # 70% personal, 30% empresa
        cuentas.append((cuenta, pais, perfil))
    return cuentas


cuentas = generar_cuentas()


# Función para seleccionar una cuenta basada en su perfil
def seleccionar_cuenta(perfil):
    cuenta, pais, _ = random.choice([cuenta for cuenta in cuentas if cuenta[2] == perfil])
    return cuenta, pais


# Función para generar una transacción aleatoria
def generar_transaccion():
    fecha = faker.date_time_this_month()
    perfil_origen, perfil_destino = random.choices(["personal", "empresa"], k=2)
    cuenta_origen, pais_procedencia = seleccionar_cuenta(perfil_origen)
    cuenta_destino, pais_remitente = seleccionar_cuenta(perfil_destino)
    monto = generar_monto(perfil_origen, perfil_destino)
    descripcion = faker.text(max_nb_chars=50)
    return [fecha, monto, descripcion, cuenta_origen, cuenta_destino, pais_procedencia, pais_remitente, perfil_origen,
            perfil_destino, 0]  # 0 indica que la transacción no es fraudulenta


# Función para generar una transacción fraudulenta
def generar_transaccion_fraudulenta():
    fecha = faker.date_time_this_month(before_now=True, after_now=False)
    descripcion = faker.text(max_nb_chars=50)

    # Patrón de fraude: uso inusual de cuentas y países
    if random.random() < 0.5:
        # Pago elevado de cuenta empresa a cuenta personal
        perfil_origen, perfil_destino = "empresa", "personal"
    else:
        # Pago elevado de cuenta personal a cuenta empresa
        perfil_origen, perfil_destino = "personal", "empresa"

    cuenta_origen, pais_procedencia = seleccionar_cuenta(perfil_origen)
    cuenta_destino, pais_remitente = seleccionar_cuenta(perfil_destino)
    monto = generar_monto(perfil_origen, perfil_destino, is_fraud=True)

    # Mayor probabilidad de que el país de destino sea de alto riesgo
    if random.random() < 0.7:
        pais_remitente = random.choice(paises_alto_riesgo)
    else:
        pais_remitente = random.choice(paises)

    return [fecha, monto, descripcion, cuenta_origen, cuenta_destino, pais_procedencia, pais_remitente, perfil_origen,
            perfil_destino, 1]  # 1 indica que la transacción es fraudulenta


# Función para generar un monto basado en el perfil de las cuentas
def generar_monto(perfil_origen, perfil_destino, is_fraud=False):
    if perfil_origen == "personal" and perfil_destino == "empresa":
        if is_fraud:
            return round(random.uniform(10000, 50000), 2)  # Transacción fraudulenta: monto inusualmente alto
        else:
            return round(random.uniform(1, 2000), 2)  # Transacción normal: pequeñas cantidades de dinero
    elif perfil_origen == "empresa" and perfil_destino == "personal":
        if is_fraud:
            return round(random.uniform(1, 100), 2)  # Transacción fraudulenta: monto inusualmente bajo
        else:
            return round(random.uniform(1000, 5000), 2)  # Transacción normal: salarios u otros pagos más altos
    else:
        return round(random.uniform(1, 10000), 2)  # Otros casos, transacciones normales con montos variados


# Función para escribir las transacciones en el archivo CSV
def escribir_transacciones(writer, num_transacciones):
    num_transacciones_fraudulentas = int(num_transacciones * 0.01)  # 1% de las transacciones son fraudulentas
    transacciones = []

    for _ in range(num_transacciones):
        if len(transacciones) < num_transacciones_fraudulentas and random.random() < 0.05:
            transaccion = generar_transaccion_fraudulenta()
        else:
            transaccion = generar_transaccion()
        transacciones.append(transaccion)

    random.shuffle(
        transacciones)  # Mezclar las transacciones para que las fraudulentas estén distribuidas aleatoriamente

    for transaccion in transacciones:
        writer.writerow(transaccion)

    return num_transacciones


# Función principal
def main():
    # Nombre del archivo CSV
    archivo_csv = 'entrenamiento.csv'

    num_transacciones_totales = 0

    # Verificamos si el archivo existe y está vacío
    if not os.path.isfile(archivo_csv) or os.stat(archivo_csv).st_size == 0:
        with open(archivo_csv, 'a', newline='') as csvfile:
            # Definimos los nombres de las columnas en el archivo CSV
            fieldnames = ['fecha', 'dinero', 'descripcion', 'cuenta_origen', 'cuenta_destino', 'pais_procedencia',
                          'pais_remitente', 'tipo_origen', 'tipo_destino', 'fraude']
            writer = csv.writer(csvfile)
            writer.writerow(fieldnames)

    # Ciclo principal para generar transacciones continuamente
    while True:
        with open(archivo_csv, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)

            num_transacciones_escritas = escribir_transacciones(writer, 100000)
            num_transacciones_totales += num_transacciones_escritas
            print(f"{num_transacciones_totales} transacciones escritas en total.")

        if num_transacciones_totales > 500000:
            break



if __name__ == "__main__":
    main()
