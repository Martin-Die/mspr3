"""
Démarrage du serveur FastAPI (uvicorn).

Usage :
    python run_server.py
    python main.py
"""

import logging
import os

import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run() -> None:
    """Lance l'API définie dans main.py."""
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "true").lower() in ("1", "true", "yes")

    logger.info("[Serveur] Démarrage de l'API sur http://%s:%d/docs", host, port)
    logger.info("[Serveur] Rechargement automatique : %s", "activé" if reload else "désactivé")
    uvicorn.run("main:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    run()
