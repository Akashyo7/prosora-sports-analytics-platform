# ⚽ Prosora Sports Analytics Platform

A comprehensive football prediction platform that combines multiple machine learning models to provide accurate match predictions, integrating AIFootballPredictions and EPL-Predictor repositories with modern web technologies.

## 🌟 Features

### 🎯 Prediction Models
- **Over/Under 2.5 Goals**: Advanced probability calculations for goal totals
- **Exact Score Predictions**: ML-powered score forecasting
- **Match Outcome Probabilities**: Win/Draw/Loss predictions with confidence scores
- **Combined Analysis**: Unified predictions from multiple models

### 🏆 Supported Leagues
- Premier League (England)
- La Liga (Spain)
- Bundesliga (Germany)
- Serie A (Italy)
- Ligue 1 (France)

### 🚀 Technology Stack
- **Backend**: FastAPI with Python 3.11+
- **Frontend**: Streamlit with modern UI components
- **Database**: Supabase (PostgreSQL)
- **ML Models**: scikit-learn, XGBoost, TensorFlow
- **Deployment**: Render
- **Integration**: n8n workflows, Flowise AI

## 📁 Project Structure

```
prosora-sports-Analytics-project/
├── AIFootballPredictions/          # Original AI Football Predictions repo
├── EPL-Predictor/                  # Original EPL Predictor repo
├── api_wrapper.py                  # Unified FastAPI backend
├── enhanced_streamlit_app.py       # Modern Streamlit dashboard
├── requirements.txt                # Python dependencies
├── supabase_schema.sql            # Database schema
├── render.yaml                    # Render deployment config
├── .env.example                   # Environment variables template
└── README.md                      # This file
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.11+
- Git
- Supabase account (optional, for data persistence)

### Local Development

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd prosora-sports-Analytics-project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your actual values
   ```

4. **Set up Supabase (Optional)**
   - Create a new Supabase project
   - Run the SQL schema from `supabase_schema.sql`
   - Update `.env` with your Supabase credentials

5. **Start the API server**
   ```bash
   uvicorn api_wrapper:app --reload --host 0.0.0.0 --port 8000
   ```

6. **Start the Streamlit dashboard**
   ```bash
   streamlit run enhanced_streamlit_app.py
   ```

## 🌐 API Endpoints

### Core Predictions
- `POST /predict/over-under` - Over/Under 2.5 goals prediction
- `POST /predict/exact-score` - Exact score prediction
- `POST /predict/combined` - Combined analysis

### Fixtures & Data
- `GET /fixtures` - Get upcoming fixtures
- `GET /health` - API health check

### Example API Usage

```python
import requests

# Get combined prediction
response = requests.post(
    "http://localhost:8000/predict/combined",
    json={
        "home_team": "Manchester United",
        "away_team": "Liverpool",
        "league_code": "E0"
    }
)

prediction = response.json()
print(f"Predicted Score: {prediction['predicted_home_goals']:.1f} - {prediction['predicted_away_goals']:.1f}")
print(f"Over 2.5 Goals: {prediction['over_25_probability']:.1%}")
```

## 🚀 Deployment

### Render Deployment

1. **Connect your GitHub repository to Render**

2. **Create services using render.yaml**
   - The configuration will automatically create both API and dashboard services

3. **Set environment variables**
   - Add your Supabase credentials
   - Configure any additional settings

4. **Deploy**
   - Services will auto-deploy on git push

### Manual Deployment

For other platforms, ensure you:
- Install Python 3.11+
- Set all required environment variables
- Run database migrations (if using Supabase)
- Start services with production settings

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `FOOTBALL_DATA_API_KEY` | Football-data.org API key | Yes |
| `SUPABASE_URL` | Supabase project URL | Optional |
| `SUPABASE_KEY` | Supabase anon key | Optional |
| `API_HOST` | API host address | No |
| `API_PORT` | API port number | No |

### League Codes

| League | Code |
|--------|------|
| Premier League | E0 |
| La Liga | SP1 |
| Bundesliga | D1 |
| Serie A | I1 |
| Ligue 1 | F1 |

## 🤖 Integration with n8n & Flowise

### n8n Workflows
The API is designed to work seamlessly with n8n:

1. **HTTP Request Node**: Call prediction endpoints
2. **Webhook Node**: Receive prediction results
3. **Database Node**: Store predictions in Supabase
4. **Schedule Node**: Automate daily predictions

### Flowise AI Integration
- Use the API as a custom tool in Flowise
- Create AI agents that can make predictions
- Build conversational interfaces for match analysis

## 📊 Database Schema

The platform includes a comprehensive database schema with tables for:
- `predictions` - Store all prediction results
- `fixtures` - Upcoming match fixtures
- `teams` - Team information and statistics
- `match_results` - Historical match data
- `prediction_accuracy` - Model performance tracking

## 🧪 Testing

```bash
# Run API tests
python -m pytest tests/

# Test specific endpoints
curl -X POST "http://localhost:8000/predict/combined" \
     -H "Content-Type: application/json" \
     -d '{"home_team": "Arsenal", "away_team": "Chelsea", "league_code": "E0"}'
```

## 📈 Model Performance

The platform combines multiple prediction models:

- **AIFootballPredictions**: Focus on over/under goals with advanced feature engineering
- **EPL-Predictor**: Specialized in exact score predictions using ensemble methods
- **Combined Model**: Weighted predictions from both models with confidence scoring

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **AIFootballPredictions**: Original over/under prediction model
- **EPL-Predictor**: Original exact score prediction model
- **Football-data.org**: Match fixtures and data API
- **Supabase**: Database and backend services
- **Render**: Deployment platform

## 📞 Support

For support and questions:
- Create an issue in this repository
- Check the [documentation](docs/)
- Review the API documentation at `/docs` when running locally

---

**Built with ❤️ for football analytics enthusiasts**