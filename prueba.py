import asyncio
import websockets
import json

async def main():
    uri = "ws://192.168.31.218:8765"
    async with websockets.connect(uri) as websocket:
        # Datos del jugador Harry Maguire
        datos_jugador = {
            "Posicion Num": 4,
            "Edad": 33,
            "Partidos Jugados": 32,
            "Goles": 27,
            "Asistencias": 12,
            "Penales Acertados": 7,
            "Tarjetas Amarillas": 0,
            "Tarjetas Rojas": 0,
            "Goles Esperados": 23.5
        }

        mensaje = json.dumps(datos_jugador)
        print(f"Enviando datos: {mensaje}")
        await websocket.send(mensaje)

        respuesta = await websocket.recv()
        print(f"Respuesta del servidor: {respuesta}")

asyncio.run(main())
