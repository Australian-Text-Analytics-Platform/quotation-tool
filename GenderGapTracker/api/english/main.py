from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pymongo import MongoClient

from db.config import config
from endpoints.outlet_stats import outlet_router

# Constants
HOST = config["MONGO_HOST"]
PORT = config["MONGO_PORT"]
MONGO_ARGS = config["MONGO_ARGS"]
DB = config["DB_NAME"]
STATIC_PATH = "gender-gap-tracker"
STATIC_HTML = "tracker.html"

app = FastAPI(
    title="Gender Gap Tracker",
    description="RESTful API for the Gender Gap Tracker public-facing dashboard",
    version="1.0.0",
)


@app.get("/", include_in_schema=False)
async def root() -> HTMLResponse:
    with open(Path(f"{STATIC_PATH}") / STATIC_HTML, "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, media_type="text/html")


@app.on_event("startup")
def startup_db_client() -> None:
    app.mongodb_client = MongoClient(HOST, PORT, **MONGO_ARGS)
    app.connection = app.mongodb_client[DB]
    print("Successfully connected to MongoDB!")


@app.on_event("shutdown")
def shutdown_db_client() -> None:
    app.mongodb_client.close()


# Attach routes
app.include_router(outlet_router, prefix="/expertWomen", tags=["info"])
# Add additional routers here for future endpoints
# ...

# Serve static files for front end from directory specified as STATIC_PATH
app.mount("/", StaticFiles(directory=STATIC_PATH), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
