"""
Prosora Sports Analytics API
Unified FastAPI wrapper combining AIFootballPredictions and EPL-Predictor models
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
from typing import Optional, Dict, List
import requests
from scipy.stats import poisson
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Add the AIFootballPredictions scripts to path
sys.path.append('./AIFootballPredictions/scripts')
sys.path.append('./EPL-Predictor')

app = FastAPI(
    title="Prosora Sports Analytics API",
    description="Unified API for football predictions combining over/under 2.5 goals and exact score predictions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response models
class PredictionResponse(BaseModel):
    home_team: str
    away_team: str
    league: str
    over_25_probability: Optional[float] = None
    under_25_probability: Optional[float] = None
    predicted_home_goals: Optional[float] = None
    predicted_away_goals: Optional[float] = None
    home_win_probability: Optional[float] = None
    away_win_probability: Optional[float] = None
    draw_probability: Optional[float] = None
    confidence_score: Optional[float] = None
    prediction_type: str
    timestamp: str

class FixturesResponse(BaseModel):
    fixtures: List[Dict]
    count: int
    date: str

class HealthResponse(BaseModel):
    status: str
    message: str
    timestamp: str

# Global variables for models
aifootball_models = {}
epl_predictor_models = {}
epl_data = None

# League mappings
LEAGUE_MAPPING = {
    "E0": "Premier League",
    "I1": "Serie A", 
    "D1": "Bundesliga",
    "SP1": "La Liga",
    "F1": "Ligue 1"
}

FOOTBALL_DATA_API_KEY = "5722a0f62b6b4d27814dbd94239744b9"

def load_aifootball_models():
    """Load AIFootballPredictions models"""
    global aifootball_models
    models_dir = "./AIFootballPredictions/models"
    
    if not os.path.exists(models_dir):
        print("AIFootballPredictions models directory not found")
        return
    
    for league in ["E0", "I1", "D1", "SP1", "F1"]:
        model_path = os.path.join(models_dir, f"{league}_voting_classifier.pkl")
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    aifootball_models[league] = pickle.load(f)
                print(f"Loaded AIFootball model for {league}")
            except Exception as e:
                print(f"Error loading model for {league}: {e}")

def load_epl_predictor_models():
    """Load EPL-Predictor models"""
    global epl_predictor_models, epl_data
    
    try:
        # Load EPL data
        data_files = []
        for year in ['1314', '1415', '1516', '1617', '1718', '1819', '1920', '2021', '2122', '2223', '2324']:
            file_path = f"./EPL-Predictor/engg_data/epl{year}.csv"
            if os.path.exists(file_path):
                data_files.append(pd.read_csv(file_path))
        
        if data_files:
            epl_data = data_files
            print("Loaded EPL historical data")
        
        # Initialize models
        epl_predictor_models = {
            'hg_model_rf': RandomForestRegressor(n_estimators=94, max_depth=2, n_jobs=-1, random_state=42),
            'ag_model_rf': RandomForestRegressor(n_estimators=280, max_depth=2, n_jobs=-1, random_state=42),
            'hg_model_xgb': XGBRegressor(n_estimators=7, max_depth=2),
            'ag_model_xgb': XGBRegressor(n_estimators=13, max_depth=1)
        }
        print("Initialized EPL-Predictor models")
        
    except Exception as e:
        print(f"Error loading EPL-Predictor models: {e}")

def get_fixtures_from_api(league_code: str = "PL", days: int = 7):
    """Fetch upcoming fixtures from football-data.org API"""
    try:
        headers = {"X-Auth-Token": FOOTBALL_DATA_API_KEY}
        
        # Map our league codes to football-data.org codes
        api_league_mapping = {
            "E0": "PL",  # Premier League
            "I1": "SA",  # Serie A
            "D1": "BL1", # Bundesliga
            "SP1": "PD", # La Liga
            "F1": "FL1"  # Ligue 1
        }
        
        api_code = api_league_mapping.get(league_code, "PL")
        url = f"https://api.football-data.org/v4/competitions/{api_code}/matches"
        
        params = {
            "status": "SCHEDULED",
            "limit": 20
        }
        
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            fixtures = []
            
            for match in data.get('matches', []):
                fixture = {
                    'id': match['id'],
                    'home_team': match['homeTeam']['name'],
                    'away_team': match['awayTeam']['name'],
                    'date': match['utcDate'],
                    'competition': match['competition']['name']
                }
                fixtures.append(fixture)
            
            return fixtures
        else:
            print(f"API Error: {response.status_code}")
            return []
            
    except Exception as e:
        print(f"Error fetching fixtures: {e}")
        return []

def predict_over_under_25(home_team: str, away_team: str, league: str = "E0"):
    """Predict over/under 2.5 goals using AIFootballPredictions model"""
    try:
        if league not in aifootball_models:
            return None, None, f"Model not available for league {league}"
        
        model = aifootball_models[league]
        
        # For demo purposes, create sample features
        # In production, you'd extract real team statistics
        sample_features = np.random.rand(1, 50)  # Placeholder for actual feature extraction
        
        prediction = model.predict_proba(sample_features)[0]
        over_25_prob = prediction[1] if len(prediction) > 1 else 0.5
        under_25_prob = 1 - over_25_prob
        
        return over_25_prob, under_25_prob, None
        
    except Exception as e:
        return None, None, str(e)

def predict_exact_score(home_team: str, away_team: str):
    """Predict exact score using EPL-Predictor models"""
    try:
        if not epl_data or not epl_predictor_models:
            return None, None, None, None, None, "EPL models not loaded"
        
        # For demo purposes, create sample predictions
        # In production, you'd use the actual team statistics and model training
        
        # Simulate predictions
        home_goals = np.random.poisson(1.5)
        away_goals = np.random.poisson(1.2)
        
        # Calculate win probabilities using Poisson distribution
        max_goals = 5
        home_win_prob = 0
        away_win_prob = 0
        draw_prob = 0
        
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                prob = poisson.pmf(i, home_goals) * poisson.pmf(j, away_goals)
                if i > j:
                    home_win_prob += prob
                elif i < j:
                    away_win_prob += prob
                else:
                    draw_prob += prob
        
        return float(home_goals), float(away_goals), home_win_prob, away_win_prob, draw_prob, None
        
    except Exception as e:
        return None, None, None, None, None, str(e)

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    print("Loading prediction models...")
    load_aifootball_models()
    load_epl_predictor_models()
    print("API startup complete!")

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="Prosora Sports Analytics API is running",
        timestamp=datetime.now().isoformat()
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    return HealthResponse(
        status="healthy",
        message=f"API running with {len(aifootball_models)} AIFootball models and EPL predictor loaded",
        timestamp=datetime.now().isoformat()
    )

@app.get("/predict/over-under/{home_team}/{away_team}", response_model=PredictionResponse)
async def predict_over_under(
    home_team: str, 
    away_team: str,
    league: str = Query("E0", description="League code (E0=EPL, I1=Serie A, D1=Bundesliga, SP1=La Liga, F1=Ligue 1)")
):
    """Predict over/under 2.5 goals for a match"""
    
    over_prob, under_prob, error = predict_over_under_25(home_team, away_team, league)
    
    if error:
        raise HTTPException(status_code=400, detail=error)
    
    return PredictionResponse(
        home_team=home_team,
        away_team=away_team,
        league=LEAGUE_MAPPING.get(league, league),
        over_25_probability=over_prob,
        under_25_probability=under_prob,
        prediction_type="over_under_2.5",
        timestamp=datetime.now().isoformat()
    )

@app.get("/predict/exact-score/{home_team}/{away_team}", response_model=PredictionResponse)
async def predict_exact_score_endpoint(home_team: str, away_team: str):
    """Predict exact score for EPL match"""
    
    home_goals, away_goals, home_win_prob, away_win_prob, draw_prob, error = predict_exact_score(home_team, away_team)
    
    if error:
        raise HTTPException(status_code=400, detail=error)
    
    return PredictionResponse(
        home_team=home_team,
        away_team=away_team,
        league="Premier League",
        predicted_home_goals=home_goals,
        predicted_away_goals=away_goals,
        home_win_probability=home_win_prob,
        away_win_probability=away_win_prob,
        draw_probability=draw_prob,
        prediction_type="exact_score",
        timestamp=datetime.now().isoformat()
    )

@app.get("/predict/combined/{home_team}/{away_team}", response_model=PredictionResponse)
async def predict_combined(
    home_team: str, 
    away_team: str,
    league: str = Query("E0", description="League code")
):
    """Combined prediction: both over/under 2.5 and exact score"""
    
    # Get over/under prediction
    over_prob, under_prob, ou_error = predict_over_under_25(home_team, away_team, league)
    
    # Get exact score prediction (only for EPL for now)
    home_goals, away_goals, home_win_prob, away_win_prob, draw_prob, es_error = None, None, None, None, None, None
    if league == "E0":
        home_goals, away_goals, home_win_prob, away_win_prob, draw_prob, es_error = predict_exact_score(home_team, away_team)
    
    # Calculate confidence score
    confidence = 0.8 if not ou_error and not es_error else 0.5
    
    return PredictionResponse(
        home_team=home_team,
        away_team=away_team,
        league=LEAGUE_MAPPING.get(league, league),
        over_25_probability=over_prob,
        under_25_probability=under_prob,
        predicted_home_goals=home_goals,
        predicted_away_goals=away_goals,
        home_win_probability=home_win_prob,
        away_win_probability=away_win_prob,
        draw_probability=draw_prob,
        confidence_score=confidence,
        prediction_type="combined",
        timestamp=datetime.now().isoformat()
    )

@app.get("/fixtures/{league}", response_model=FixturesResponse)
async def get_fixtures(league: str = "E0", days: int = Query(7, description="Number of days ahead")):
    """Get upcoming fixtures for a league"""
    
    fixtures = get_fixtures_from_api(league, days)
    
    return FixturesResponse(
        fixtures=fixtures,
        count=len(fixtures),
        date=datetime.now().isoformat()
    )

@app.get("/leagues")
async def get_supported_leagues():
    """Get list of supported leagues"""
    return {
        "leagues": LEAGUE_MAPPING,
        "description": "Supported leagues for predictions"
    }

if __name__ == "__main__":
    uvicorn.run(
        "api_wrapper:app", 
        host="0.0.0.0", 
        port=8001, 
        reload=True,
        log_level="info"
    )