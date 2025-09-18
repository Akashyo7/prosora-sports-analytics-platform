-- Prosora Sports Analytics Database Schema
-- Supabase SQL schema for storing predictions and match data

-- Create predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    league TEXT NOT NULL,
    league_code TEXT,
    
    -- Over/Under 2.5 Goals Predictions
    over_25_probability FLOAT,
    under_25_probability FLOAT,
    
    -- Exact Score Predictions
    predicted_home_goals FLOAT,
    predicted_away_goals FLOAT,
    
    -- Win Probabilities
    home_win_probability FLOAT,
    away_win_probability FLOAT,
    draw_probability FLOAT,
    
    -- Metadata
    confidence_score FLOAT,
    prediction_type TEXT NOT NULL, -- 'over_under', 'exact_score', 'combined'
    
    -- Match Information
    match_date TIMESTAMP,
    fixture_id TEXT,
    
    -- Actual Results (filled after match completion)
    actual_home_goals INTEGER,
    actual_away_goals INTEGER,
    actual_result TEXT, -- 'H', 'A', 'D'
    actual_over_25 BOOLEAN,
    
    -- Accuracy Tracking
    over_25_correct BOOLEAN,
    exact_score_correct BOOLEAN,
    result_correct BOOLEAN,
    
    -- API and Processing Info
    prediction_json JSONB,
    api_version TEXT DEFAULT '1.0.0',
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create fixtures table for upcoming matches
CREATE TABLE IF NOT EXISTS fixtures (
    id SERIAL PRIMARY KEY,
    fixture_id TEXT UNIQUE NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    league TEXT NOT NULL,
    league_code TEXT,
    match_date TIMESTAMP NOT NULL,
    status TEXT DEFAULT 'SCHEDULED', -- 'SCHEDULED', 'LIVE', 'FINISHED', 'CANCELLED'
    
    -- External API data
    api_data JSONB,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create teams table for team information and statistics
CREATE TABLE IF NOT EXISTS teams (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    league TEXT NOT NULL,
    league_code TEXT,
    
    -- Team Statistics (updated regularly)
    matches_played INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0,
    draws INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    goals_for INTEGER DEFAULT 0,
    goals_against INTEGER DEFAULT 0,
    
    -- Form and Performance Metrics
    recent_form TEXT, -- Last 5 matches: 'WWDLL'
    avg_goals_scored FLOAT DEFAULT 0,
    avg_goals_conceded FLOAT DEFAULT 0,
    over_25_percentage FLOAT DEFAULT 0,
    
    -- Timestamps
    last_updated TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create match_results table for historical data
CREATE TABLE IF NOT EXISTS match_results (
    id SERIAL PRIMARY KEY,
    fixture_id TEXT UNIQUE,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    league TEXT NOT NULL,
    league_code TEXT,
    
    -- Match Results
    home_goals INTEGER NOT NULL,
    away_goals INTEGER NOT NULL,
    result TEXT NOT NULL, -- 'H', 'A', 'D'
    total_goals INTEGER GENERATED ALWAYS AS (home_goals + away_goals) STORED,
    over_25 BOOLEAN GENERATED ALWAYS AS (home_goals + away_goals > 2) STORED,
    
    -- Match Details
    match_date TIMESTAMP NOT NULL,
    season TEXT,
    matchweek INTEGER,
    
    -- Additional Statistics (if available)
    home_shots INTEGER,
    away_shots INTEGER,
    home_shots_on_target INTEGER,
    away_shots_on_target INTEGER,
    
    -- External API data
    api_data JSONB,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create prediction_accuracy table for tracking model performance
CREATE TABLE IF NOT EXISTS prediction_accuracy (
    id SERIAL PRIMARY KEY,
    model_type TEXT NOT NULL, -- 'over_under', 'exact_score', 'combined'
    league TEXT NOT NULL,
    
    -- Accuracy Metrics
    total_predictions INTEGER DEFAULT 0,
    correct_predictions INTEGER DEFAULT 0,
    accuracy_percentage FLOAT GENERATED ALWAYS AS (
        CASE 
            WHEN total_predictions > 0 THEN (correct_predictions::FLOAT / total_predictions * 100)
            ELSE 0 
        END
    ) STORED,
    
    -- Time Period
    date_from DATE NOT NULL,
    date_to DATE NOT NULL,
    
    -- Detailed Metrics (JSONB for flexibility)
    metrics JSONB,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_predictions_teams ON predictions(home_team, away_team);
CREATE INDEX IF NOT EXISTS idx_predictions_league ON predictions(league_code);
CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_predictions_type ON predictions(prediction_type);

CREATE INDEX IF NOT EXISTS idx_fixtures_date ON fixtures(match_date);
CREATE INDEX IF NOT EXISTS idx_fixtures_league ON fixtures(league_code);
CREATE INDEX IF NOT EXISTS idx_fixtures_status ON fixtures(status);

CREATE INDEX IF NOT EXISTS idx_teams_league ON teams(league_code);
CREATE INDEX IF NOT EXISTS idx_teams_name ON teams(name);

CREATE INDEX IF NOT EXISTS idx_match_results_date ON match_results(match_date);
CREATE INDEX IF NOT EXISTS idx_match_results_league ON match_results(league_code);
CREATE INDEX IF NOT EXISTS idx_match_results_teams ON match_results(home_team, away_team);

-- Create functions for updating timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for automatic timestamp updates
CREATE TRIGGER update_predictions_updated_at BEFORE UPDATE ON predictions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_fixtures_updated_at BEFORE UPDATE ON fixtures
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_prediction_accuracy_updated_at BEFORE UPDATE ON prediction_accuracy
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create views for common queries
CREATE OR REPLACE VIEW recent_predictions AS
SELECT 
    p.*,
    f.match_date,
    f.status as match_status
FROM predictions p
LEFT JOIN fixtures f ON (p.home_team = f.home_team AND p.away_team = f.away_team)
WHERE p.created_at >= NOW() - INTERVAL '7 days'
ORDER BY p.created_at DESC;

CREATE OR REPLACE VIEW league_accuracy AS
SELECT 
    league,
    prediction_type,
    COUNT(*) as total_predictions,
    COUNT(*) FILTER (WHERE over_25_correct = true OR exact_score_correct = true OR result_correct = true) as correct_predictions,
    ROUND(
        (COUNT(*) FILTER (WHERE over_25_correct = true OR exact_score_correct = true OR result_correct = true)::FLOAT / COUNT(*) * 100), 2
    ) as accuracy_percentage
FROM predictions 
WHERE actual_home_goals IS NOT NULL 
GROUP BY league, prediction_type
ORDER BY accuracy_percentage DESC;

-- Insert sample data for testing (optional)
-- INSERT INTO teams (name, league, league_code) VALUES 
-- ('Manchester United', 'Premier League', 'E0'),
-- ('Liverpool', 'Premier League', 'E0'),
-- ('Manchester City', 'Premier League', 'E0'),
-- ('Arsenal', 'Premier League', 'E0');

-- Grant permissions (adjust as needed for your Supabase setup)
-- GRANT ALL ON ALL TABLES IN SCHEMA public TO authenticated;
-- GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO authenticated;