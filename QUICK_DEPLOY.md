## Quick Deployment Guide

### 1. Deploy Backend to Render

1. Push your code to GitHub
2. Go to https://render.com → New → Web Service
3. Connect your repository
4. Settings:
   - **Root Directory**: `backend`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Deploy and copy your backend URL (e.g., `https://cricket-predictor-api.onrender.com`)

> ⚡ **Fast deployment**: Pre-trained model included in repo - no training needed!

### 2. Deploy Frontend to Vercel

1. Go to https://vercel.com → New Project
2. Import your repository
3. Add Environment Variable:
   - Name: `VITE_API_URL`
   - Value: Your Render backend URL (from step 1)
4. Deploy!

### 3. Test Your App

- Visit your Vercel URL
- Try making a prediction
- Check browser console for any errors

**Full guide**: See [DEPLOYMENT.md](./DEPLOYMENT.md)
