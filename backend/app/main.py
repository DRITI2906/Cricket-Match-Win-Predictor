from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import prediction

app = FastAPI(title="Win Wise Cricket Insight API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (you can restrict this later)
    allow_credentials=False,  # Must be False when using allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(prediction.router, prefix="/api", tags=["prediction"])

@app.get("/")
async def root():
    return {"message": "Win Wise Cricket Insight API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
