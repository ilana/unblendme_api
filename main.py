from fastapi import FastAPI
from app.routes import router
from app.config import load_env

# Load environment variables
load_env()

app = FastAPI(title="Parts Classification API")

# Include API routes
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
