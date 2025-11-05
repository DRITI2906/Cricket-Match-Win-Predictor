from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import prediction

app = FastAPI(title="Win Wise Cricket Insight API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Default Vite dev server
        "http://localhost:8080",  # Alternative Vite port
        "http://localhost:3000",  # Common React port
        "https://*.vercel.app",  # All Vercel deployments
        "https://vercel.app",    # Vercel domains
    ],
    allow_credentials=True,
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
