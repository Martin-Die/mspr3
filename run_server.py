"""
Démarrage du serveur FastAPI (uvicorn).

Usage :
    python run_server.py
    python main.py
"""

import os

import uvicorn


def run() -> None:
    """Lance l'API définie dans main.py."""
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "true").lower() in ("1", "true", "yes")

    print(f"API disponible sur http://{host}:{port}/docs")
    uvicorn.run("main:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    run()
