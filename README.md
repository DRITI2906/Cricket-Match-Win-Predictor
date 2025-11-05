# 🏏 Cricket Match Win Predictor

An AI-powered Cricket Match Win Probability Predictor that estimates the likelihood of a batting team winning a match based on live game data — built with Machine Learning (Random Forest), FastAPI, and React.
The project also integrates Explainable AI (SHAP Analysis) to make model decisions transparent and interpretable.

- **Deployed Link** - https://cricket-match-win-predictor.vercel.app/

## 🚀 Project Overview

This web application predicts the winning probability of a team during a live cricket match scenario.
The user can input key parameters like:

 - Runs required

- Balls remaining

- Wickets in hand

- Current run rate

Based on these inputs, the trained Random Forest model calculates the real-time win probability and visualizes how each factor contributes to the outcome using SHAP (SHapley Additive exPlanations).

## 🧩 Features

- Real-time win prediction
- Interactive dashboard with clean UI
- Explainable AI via SHAP
- FastAPI backend for high-speed inference
- Modular ML pipeline (train → serve → visualize)
- Deployed-ready full-stack structure

## 🖥️ Tech Stack

### Backend — FastAPI

- Handles prediction requests

- Loads the trained Random Forest model

- Returns win probability and SHAP explanations in JSON

- High-speed asynchronous API framework

### Frontend — React.js

- Displays win probabilities dynamically

- Shows visual progress bars and confidence indicators

- Integrates SHAP visualization for model interpretability

### ML Explainability — SHAP

- Enhances transparency in model predictions

- Displays contribution of each feature in the decision-making process

## 🧰 Installation & Setup 

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/DRITI2906/Cricket-Match-Win-Predictor.git
   cd Cricket-Match-Win-Predictor
   ```
2. **Backend Setup (FastAPI)**


   ```bash
   cd backend
   pip install -r requirements.txt
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

3. **Frontend Setup (React)**


   ```bash
   npm install
   npm run dev
   ```
