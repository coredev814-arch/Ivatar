from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes.avatar import router as avatar_router
from .api.routes.garment import router as garment_router


def create_app() -> FastAPI:
    app = FastAPI(title="SMPL Avatar Service", version="0.1.0")

    # Allow the Vite dev server and common local origins; tighten in production.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", tags=["system"])
    def health_check():
        return {"status": "ok"}

    app.include_router(avatar_router, prefix="/api/avatar", tags=["avatar"])
    app.include_router(garment_router, prefix="/api/garments", tags=["garments"])

    return app


app = create_app()

