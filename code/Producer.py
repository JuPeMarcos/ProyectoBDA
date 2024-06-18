import time
import random
from faker import Faker
import mysql.connector

# Configuración de la conexión a la base de datos
config = {
    'host': '192.168.137.8',
    'user': 'root',
    'password': 'usuario',
    'database': 'transacciones_bancarias',
    'port': 3306
}
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
def generar_cuentas(num_cuentas=10000):
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

# Función para generar una transacción aleatoria
def generar_transaccion():
    fecha = faker.date_time_this_month()
    perfil_origen, perfil_destino = random.choices(["personal", "empresa"], k=2)
    cuenta_origen, pais_procedencia = seleccionar_cuenta(perfil_origen)
    cuenta_destino, pais_remitente = seleccionar_cuenta(perfil_destino)
    monto = generar_monto(perfil_origen, perfil_destino)
    descripcion = faker.text(max_nb_chars=50)
    return [fecha, monto, descripcion, cuenta_origen, cuenta_destino, pais_procedencia, pais_remitente, perfil_origen, perfil_destino]

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

    return [fecha, monto, descripcion, cuenta_origen, cuenta_destino, pais_procedencia, pais_remitente, perfil_origen, perfil_destino]

def insertar_transaccion(cursor, transaccion):
    cursor.execute("""
        INSERT INTO transacciones (fecha, dinero, descripcion, cuenta_origen, cuenta_destino, pais_procedencia, pais_remitente, tipo_origen, tipo_destino)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, transaccion)

def main():
    num_transacciones_totales = 0
    try:
        while True:
            # Conectar a la base de datos
            cnx = mysql.connector.connect(**config)
            cursor = cnx.cursor()

            for _ in range(1000):
                # Generar una transacción aleatoria o fraudulenta
                if random.random() < 0.01:  # 1% de probabilidad de ser fraudulenta
                    transaccion = generar_transaccion_fraudulenta()
                else:
                    transaccion = generar_transaccion()

                # Insertar la transacción en la base de datos
                insertar_transaccion(cursor, transaccion)

                # Confirmar los cambios
                cnx.commit()
                num_transacciones_totales += 1
                print(f"{num_transacciones_totales} transacciones escritas en total.")

                # Esperar 1 segundo antes de generar la siguiente transacción
                time.sleep(0.2)

            # Cerrar la conexión
            cursor.close()
            cnx.close()

            # Preguntar al usuario si desea continuar o parar cada 10 transacciones
            respuesta = input("¿Desea continuar generando transacciones? (s/n): ")
            if respuesta.lower() != 's':
                break

    except mysql.connector.Error as error:
        print("Error al insertar datos en la tabla transacciones:", error)
    finally:
        if 'cnx' in locals() and cnx.is_connected():
            cursor.close()
            cnx.close()
            print("Conexión a MySQL cerrada.")

if __name__ == "__main__":
    main()