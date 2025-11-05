# Deployment Guide

## Backend Deployment (Render)

### Step 1: Prepare Your Repository
1. Push your code to GitHub
2. Make sure `backend/requirements.txt` and `cricket_features.csv` are committed

### Step 2: Deploy on Render
1. Go to [render.com](https://render.com) and sign in
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Configure the service:
   - **Name**: `cricket-predictor-api`
   - **Region**: Choose closest to your users
   - **Branch**: `main`
   - **Root Directory**: `backend`
   - **Runtime**: `Python 3`
   - **Build Command**: 
     ```
     pip install -r requirements.txt
     ```
   - **Start Command**: 
     ```
     uvicorn app.main:app --host 0.0.0.0 --port $PORT
     ```
   - **Instance Type**: Free (or choose paid for better performance)

> **Note**: The trained model files (`*.pkl`) are included in the repository, so no training is needed during deployment. This makes deployment faster and more reliable!

5. Add Environment Variables (if needed):
   - Click "Environment" tab
   - Add any environment variables your app needs

6. Click "Create Web Service"
7. Wait for deployment to complete (may take 5-10 minutes due to model training)
8. Note your backend URL: `https://cricket-predictor-api.onrender.com`

### Step 3: Important - CORS Configuration
Once deployed, your backend URL will be something like:
`https://cricket-predictor-api.onrender.com`

Make sure this is added to your CORS origins in `backend/app/main.py` (already configured for Vercel domains).

---

## Frontend Deployment (Vercel)

### Step 1: Update Environment Variable
1. Create a production environment file (or use Vercel dashboard):
   - You'll set `VITE_API_URL` to your Render backend URL

### Step 2: Deploy on Vercel
1. Go to [vercel.com](https://vercel.com) and sign in
2. Click "Add New" → "Project"
3. Import your GitHub repository
4. Configure the project:
   - **Framework Preset**: Vite
   - **Root Directory**: `./` (leave as root)
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
   - **Install Command**: `npm install`

5. Add Environment Variables:
   - Click "Environment Variables"
   - Add: `VITE_API_URL` = `https://your-backend-url.onrender.com`
   - Example: `VITE_API_URL=https://cricket-predictor-api.onrender.com`

6. Click "Deploy"
7. Wait for deployment to complete (2-3 minutes)
8. Your app will be live at: `https://your-project.vercel.app`

### Step 3: Update CORS in Backend
After getting your Vercel URL, update the CORS origins in `backend/app/main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:8080",
        "http://localhost:3000",
        "https://your-project.vercel.app",  # Add your Vercel URL
        "https://*.vercel.app"  # Allow all Vercel preview deployments
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Commit and push this change. Render will automatically redeploy.

---

## Post-Deployment Checklist

### Backend (Render)
- ✅ Service is running (check Render dashboard)
- ✅ Health check passes: Visit `https://your-backend.onrender.com/health`
- ✅ API is accessible: Visit `https://your-backend.onrender.com/docs` (FastAPI Swagger UI)
- ✅ Model trained successfully (check deployment logs)

### Frontend (Vercel)
- ✅ Site is live and loads correctly
- ✅ Environment variable `VITE_API_URL` is set correctly
- ✅ Can make predictions (test the form)
- ✅ No CORS errors in browser console

---

## Common Issues & Solutions

### Issue 1: "Model file not found" error on Render
**Solution**: The trained model files are included in the repository
- Verify `backend/models/cricket_model.pkl` exists in your repo
- Check Render logs to ensure files were copied during deployment
- If missing, retrain locally and commit: `python train_model.py && git add backend/models/*.pkl && git commit -m "Update model" && git push`

### Issue 2: CORS errors in browser
**Solution**: 
- Add your Vercel domain to CORS origins in `backend/app/main.py`
- Redeploy backend after updating

### Issue 3: Backend responds with 502/503 errors
**Solution**:
- Render free tier sleeps after inactivity (takes ~30 seconds to wake up)
- Consider upgrading to a paid plan for always-on service
- Or add a health check ping service (like UptimeRobot)

### Issue 4: Frontend shows "Failed to fetch"
**Solution**:
- Check `VITE_API_URL` environment variable in Vercel
- Verify backend is running on Render
- Check browser console for exact error

---

## Continuous Deployment

Both platforms support automatic deployment:

- **Render**: Automatically redeploys when you push to GitHub (main branch)
- **Vercel**: Automatically redeploys on every push (creates preview deployments for PRs)

To manually trigger deployment:
- **Render**: Go to dashboard → "Manual Deploy" → "Clear build cache & deploy"
- **Vercel**: Go to dashboard → "Deployments" → "Redeploy"

---

## Monitoring

### Backend Logs (Render)
- Go to your service dashboard
- Click "Logs" tab
- Monitor for errors or performance issues

### Frontend Logs (Vercel)
- Go to your project dashboard  
- Click "Deployments" → Select deployment → "Function Logs"
- Use browser DevTools console for client-side issues

---

## Cost Estimates

### Free Tier (Recommended for testing)
- **Render**: 750 hours/month free (sleeps after 15 min inactivity)
- **Vercel**: Unlimited bandwidth for personal projects
- **Total**: $0/month

### Paid Tier (For production)
- **Render Starter**: $7/month (always-on, better performance)
- **Vercel Pro**: $20/month (if you need team features)
- **Total**: ~$7-27/month

---

## Need Help?

- **Render Docs**: https://render.com/docs
- **Vercel Docs**: https://vercel.com/docs
- **FastAPI Deployment**: https://fastapi.tiangolo.com/deployment/
- **Vite Deployment**: https://vitejs.dev/guide/static-deploy.html

Good luck with your deployment! 🚀
