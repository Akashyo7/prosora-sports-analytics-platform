import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import os
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure Streamlit page
st.set_page_config(
    page_title="Prosora Sports Analytics Platform",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    .team-vs {
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
    }
    
    .confidence-high { color: #28a745; }
    .confidence-medium { color: #ffc107; }
    .confidence-low { color: #dc3545; }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://prosora-sports-api.onrender.com")  # Default to Render deployment

class PredictionAPI:
    def __init__(self, base_url):
        self.base_url = base_url
    
    def get_fixtures(self, league_code="E0", days_ahead=7):
        """Get upcoming fixtures"""
        try:
            response = requests.get(f"{self.base_url}/fixtures/{league_code}?days={days_ahead}")
            if response.status_code == 200:
                return response.json()
            return {"fixtures": [], "count": 0}
        except Exception as e:
            st.error(f"Error fetching fixtures: {str(e)}")
            return {"fixtures": [], "count": 0}
    
    def get_over_under_prediction(self, home_team, away_team, league_code="E0"):
        """Get over/under 2.5 goals prediction"""
        try:
            response = requests.get(f"{self.base_url}/predict/over-under/{home_team}/{away_team}?league={league_code}")
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            st.error(f"Error getting over/under prediction: {str(e)}")
            return None
    
    def get_exact_score_prediction(self, home_team, away_team, league_code="E0"):
        """Get exact score prediction"""
        try:
            response = requests.get(f"{self.base_url}/predict/exact-score/{home_team}/{away_team}?league={league_code}")
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            st.error(f"Error getting exact score prediction: {str(e)}")
            return None
    
    def get_combined_prediction(self, home_team, away_team, league_code="E0"):
        """Get combined prediction"""
        try:
            response = requests.get(f"{self.base_url}/predict/combined/{home_team}/{away_team}?league={league_code}")
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            st.error(f"Error getting combined prediction: {str(e)}")
            return None

# Initialize API client
api = PredictionAPI(API_BASE_URL)

def display_confidence_badge(confidence):
    """Display confidence level with color coding"""
    if confidence >= 0.8:
        return f'<span class="confidence-high">High ({confidence:.1%})</span>'
    elif confidence >= 0.6:
        return f'<span class="confidence-medium">Medium ({confidence:.1%})</span>'
    else:
        return f'<span class="confidence-low">Low ({confidence:.1%})</span>'

def create_probability_chart(probabilities, labels, title):
    """Create a probability visualization chart"""
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=probabilities,
            marker_color=['#667eea', '#764ba2', '#f093fb'],
            text=[f'{p:.1%}' for p in probabilities],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Outcome",
        yaxis_title="Probability",
        yaxis=dict(tickformat='.0%'),
        height=400,
        showlegend=False
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">⚽ Prosora Sports Analytics Platform</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎯 Prediction Settings")
    
    # League selection
    league_options = {
        "Premier League": "E0",
        "La Liga": "SP1",
        "Bundesliga": "D1",
        "Serie A": "I1",
        "Ligue 1": "F1"
    }
    
    selected_league = st.sidebar.selectbox(
        "Select League",
        options=list(league_options.keys()),
        index=0
    )
    league_code = league_options[selected_league]
    
    # Prediction mode
    prediction_mode = st.sidebar.radio(
        "Prediction Mode",
        ["Upcoming Fixtures", "Custom Match", "Batch Analysis"]
    )
    
    # Main content area
    if prediction_mode == "Upcoming Fixtures":
        st.header(f"📅 Upcoming {selected_league} Fixtures")
        
        # Get fixtures
        with st.spinner("Loading fixtures..."):
            fixtures_data = api.get_fixtures(league_code=league_code, days_ahead=7)
        
        # Extract fixtures list from the response
        if fixtures_data and isinstance(fixtures_data, dict) and 'fixtures' in fixtures_data:
            fixtures = fixtures_data['fixtures']
        elif fixtures_data and isinstance(fixtures_data, list):
            fixtures = fixtures_data
        else:
            fixtures = []
        
        if fixtures:
            # Display fixtures with predictions
            for fixture in fixtures[:5]:  # Show first 5 fixtures
                with st.expander(f"{fixture['home_team']} vs {fixture['away_team']} - {fixture['date']}", expanded=True):
                    col1, col2, col3 = st.columns([1, 2, 1])
                    
                    with col1:
                        st.markdown(f"**{fixture['home_team']}**")
                        st.markdown("🏠 Home")
                    
                    with col2:
                        st.markdown('<div class="team-vs">VS</div>', unsafe_allow_html=True)
                        
                        # Get combined prediction
                        prediction = api.get_combined_prediction(
                            fixture['home_team'], 
                            fixture['away_team'], 
                            league_code
                        )
                        
                        if prediction:
                            # Display key metrics
                            col_a, col_b, col_c = st.columns(3)
                            
                            with col_a:
                                over_25_prob = prediction.get('over_25_probability', 0)
                                over_25_display = f"{over_25_prob:.1%}" if over_25_prob is not None else "N/A"
                                st.metric(
                                    "Over 2.5 Goals",
                                    over_25_display,
                                    delta=None
                                )
                            
                            with col_b:
                                home_goals = prediction.get('predicted_home_goals', 0)
                                away_goals = prediction.get('predicted_away_goals', 0)
                                if home_goals is not None and away_goals is not None:
                                    score_display = f"{home_goals:.1f} - {away_goals:.1f}"
                                else:
                                    score_display = "N/A"
                                st.metric(
                                    "Predicted Score",
                                    score_display,
                                    delta=None
                                )
                            
                            with col_c:
                                confidence = prediction.get('confidence_score', 0)
                                confidence_display = f"{confidence:.1%}" if confidence is not None else "N/A"
                                st.metric(
                                    "Confidence",
                                    confidence_display,
                                    delta=None
                                )
                            
                            # Win probabilities chart
                            if all(key in prediction for key in ['home_win_probability', 'draw_probability', 'away_win_probability']):
                                probs = [
                                    prediction['home_win_probability'],
                                    prediction['draw_probability'],
                                    prediction['away_win_probability']
                                ]
                                labels = ['Home Win', 'Draw', 'Away Win']
                                
                                fig = create_probability_chart(probs, labels, "Win Probabilities")
                                st.plotly_chart(fig, use_container_width=True)
                    
                    with col3:
                        st.markdown(f"**{fixture['away_team']}**")
                        st.markdown("✈️ Away")
        else:
            st.warning("No fixtures available for the selected league.")
    
    elif prediction_mode == "Custom Match":
        st.header("🎲 Custom Match Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            home_team = st.text_input("Home Team", placeholder="e.g., Manchester United")
        
        with col2:
            away_team = st.text_input("Away Team", placeholder="e.g., Liverpool")
        
        if st.button("🔮 Generate Prediction", type="primary"):
            if home_team and away_team:
                with st.spinner("Generating predictions..."):
                    # Get all prediction types
                    over_under = api.get_over_under_prediction(home_team, away_team, league_code)
                    exact_score = api.get_exact_score_prediction(home_team, away_team, league_code)
                    combined = api.get_combined_prediction(home_team, away_team, league_code)
                
                # Display results in tabs
                tab1, tab2, tab3 = st.tabs(["🎯 Combined Analysis", "⚽ Goals Prediction", "🏆 Match Outcome"])
                
                with tab1:
                    if combined:
                        st.markdown(f'<div class="team-vs">{home_team} vs {away_team}</div>', unsafe_allow_html=True)
                        
                        # Key metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Predicted Score", f"{combined['predicted_home_goals']:.1f} - {combined['predicted_away_goals']:.1f}")
                        
                        with col2:
                            st.metric("Over 2.5 Goals", f"{combined['over_25_probability']:.1%}")
                        
                        with col3:
                            st.metric("Under 2.5 Goals", f"{combined['under_25_probability']:.1%}")
                        
                        with col4:
                            confidence_html = display_confidence_badge(combined['confidence_score'])
                            st.markdown(f"**Confidence:** {confidence_html}", unsafe_allow_html=True)
                        
                        # Win probabilities
                        if all(key in combined for key in ['home_win_probability', 'draw_probability', 'away_win_probability']):
                            probs = [
                                combined['home_win_probability'],
                                combined['draw_probability'],
                                combined['away_win_probability']
                            ]
                            labels = [f'{home_team} Win', 'Draw', f'{away_team} Win']
                            
                            fig = create_probability_chart(probs, labels, "Match Outcome Probabilities")
                            st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    if over_under:
                        st.subheader("Goals Analysis")
                        
                        # Over/Under visualization
                        ou_probs = [over_under['over_25_probability'], over_under['under_25_probability']]
                        ou_labels = ['Over 2.5', 'Under 2.5']
                        
                        fig = create_probability_chart(ou_probs, ou_labels, "Over/Under 2.5 Goals")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Additional insights
                        st.info(f"**Recommendation:** {'Over 2.5 Goals' if over_under['over_25_probability'] > 0.5 else 'Under 2.5 Goals'}")
                
                with tab3:
                    if exact_score:
                        st.subheader("Exact Score Prediction")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric(f"{home_team} Goals", f"{exact_score['predicted_home_goals']:.1f}")
                        
                        with col2:
                            st.metric(f"{away_team} Goals", f"{exact_score['predicted_away_goals']:.1f}")
                        
                        # Score range visualization
                        home_goals = exact_score['predicted_home_goals']
                        away_goals = exact_score['predicted_away_goals']
                        
                        st.markdown("### Most Likely Score Ranges")
                        
                        # Create score probability matrix (simplified)
                        scores = []
                        for h in range(0, 5):
                            for a in range(0, 5):
                                # Simple probability based on predicted goals
                                prob = np.exp(-abs(h - home_goals)) * np.exp(-abs(a - away_goals))
                                scores.append({'Home': h, 'Away': a, 'Probability': prob})
                        
                        scores_df = pd.DataFrame(scores)
                        scores_df['Probability'] = scores_df['Probability'] / scores_df['Probability'].sum()
                        top_scores = scores_df.nlargest(5, 'Probability')
                        
                        for _, score in top_scores.iterrows():
                            st.write(f"**{int(score['Home'])}-{int(score['Away'])}**: {score['Probability']:.1%}")
            else:
                st.error("Please enter both team names.")
    
    elif prediction_mode == "Batch Analysis":
        st.header("📊 Batch Analysis")
        
        st.info("Upload a CSV file with columns: home_team, away_team, league_code")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            if all(col in df.columns for col in ['home_team', 'away_team']):
                if 'league_code' not in df.columns:
                    df['league_code'] = league_code
                
                st.write(f"Processing {len(df)} matches...")
                
                # Process predictions
                results = []
                progress_bar = st.progress(0)
                
                for idx, row in df.iterrows():
                    prediction = api.get_combined_prediction(
                        row['home_team'], 
                        row['away_team'], 
                        row.get('league_code', league_code)
                    )
                    
                    if prediction:
                        results.append({
                            'Home Team': row['home_team'],
                            'Away Team': row['away_team'],
                            'Predicted Score': f"{prediction['predicted_home_goals']:.1f} - {prediction['predicted_away_goals']:.1f}",
                            'Over 2.5 Probability': f"{prediction['over_25_probability']:.1%}",
                            'Home Win Probability': f"{prediction.get('home_win_probability', 0):.1%}",
                            'Draw Probability': f"{prediction.get('draw_probability', 0):.1%}",
                            'Away Win Probability': f"{prediction.get('away_win_probability', 0):.1%}",
                            'Confidence': f"{prediction['confidence_score']:.1%}"
                        })
                    
                    progress_bar.progress((idx + 1) / len(df))
                
                # Display results
                if results:
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            else:
                st.error("CSV file must contain 'home_team' and 'away_team' columns.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>⚽ Prosora Sports Analytics Platform | Powered by AI & Machine Learning</p>
            <p>Combining AIFootballPredictions & EPL-Predictor models for comprehensive match analysis</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()