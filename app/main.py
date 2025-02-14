from fastapi import FastAPI
from app.routes import router
from app.config import load_env
import os

# Load environment variables
load_env()

app = FastAPI(title="Parts Classification API")

# Include API routes
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Use Heroku's assigned port or default to 8000
    uvicorn.run(app, host="0.0.0.0", port=port)
