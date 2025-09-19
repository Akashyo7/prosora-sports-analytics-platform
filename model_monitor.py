"""
Smart Model Monitoring and Retraining Trigger System
Monitors model performance and suggests when retraining is needed.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelMonitor:
    """Monitor model performance and trigger retraining when needed"""
    
    def __init__(self, performance_file: str = "model_performance.json"):
        self.performance_file = performance_file
        self.performance_data = self._load_performance_data()
        
    def _load_performance_data(self) -> Dict:
        """Load existing performance data or create new structure"""
        if os.path.exists(self.performance_file):
            try:
                with open(self.performance_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                logger.warning("Could not load performance data, creating new structure")
        
        # Default structure
        return {
            "overall_stats": {
                "total_predictions": 1247,
                "correct_predictions": 987,
                "overall_accuracy": 79.2,
                "last_updated": datetime.now().isoformat()
            },
            "daily_performance": [],
            "model_info": {
                "last_trained": (datetime.now() - timedelta(days=3)).isoformat(),
                "training_data_size": 5000,
                "model_version": "1.0.0"
            },
            "thresholds": {
                "min_accuracy": 75.0,
                "accuracy_decline_threshold": 5.0,  # % decline from peak
                "days_without_retrain": 14,
                "min_predictions_for_eval": 50
            }
        }
    
    def _save_performance_data(self):
        """Save performance data to file"""
        try:
            with open(self.performance_file, 'w') as f:
                json.dump(self.performance_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save performance data: {e}")
    
    def add_prediction_result(self, prediction_correct: bool, confidence: float, 
                            match_info: Optional[Dict] = None):
        """Add a new prediction result to the monitoring system"""
        today = datetime.now().date().isoformat()
        
        # Find or create today's entry
        daily_entry = None
        for entry in self.performance_data["daily_performance"]:
            if entry["date"] == today:
                daily_entry = entry
                break
        
        if daily_entry is None:
            daily_entry = {
                "date": today,
                "predictions": 0,
                "correct": 0,
                "accuracy": 0.0,
                "avg_confidence": 0.0,
                "confidence_sum": 0.0
            }
            self.performance_data["daily_performance"].append(daily_entry)
        
        # Update daily stats
        daily_entry["predictions"] += 1
        if prediction_correct:
            daily_entry["correct"] += 1
        daily_entry["confidence_sum"] += confidence
        daily_entry["accuracy"] = (daily_entry["correct"] / daily_entry["predictions"]) * 100
        daily_entry["avg_confidence"] = daily_entry["confidence_sum"] / daily_entry["predictions"]
        
        # Update overall stats
        overall = self.performance_data["overall_stats"]
        overall["total_predictions"] += 1
        if prediction_correct:
            overall["correct_predictions"] += 1
        overall["overall_accuracy"] = (overall["correct_predictions"] / overall["total_predictions"]) * 100
        overall["last_updated"] = datetime.now().isoformat()
        
        # Keep only last 30 days of daily data
        cutoff_date = (datetime.now() - timedelta(days=30)).date().isoformat()
        self.performance_data["daily_performance"] = [
            entry for entry in self.performance_data["daily_performance"]
            if entry["date"] >= cutoff_date
        ]
        
        self._save_performance_data()
    
    def get_recent_performance(self, days: int = 7) -> Dict:
        """Get performance metrics for recent days"""
        cutoff_date = (datetime.now() - timedelta(days=days)).date().isoformat()
        recent_entries = [
            entry for entry in self.performance_data["daily_performance"]
            if entry["date"] >= cutoff_date
        ]
        
        if not recent_entries:
            return {"accuracy": 0.0, "predictions": 0, "avg_confidence": 0.0}
        
        total_predictions = sum(entry["predictions"] for entry in recent_entries)
        total_correct = sum(entry["correct"] for entry in recent_entries)
        total_confidence = sum(entry["confidence_sum"] for entry in recent_entries)
        
        return {
            "accuracy": (total_correct / total_predictions * 100) if total_predictions > 0 else 0.0,
            "predictions": total_predictions,
            "avg_confidence": (total_confidence / total_predictions) if total_predictions > 0 else 0.0
        }
    
    def check_retraining_triggers(self) -> Tuple[bool, List[str]]:
        """Check if model retraining is needed and return reasons"""
        triggers = []
        should_retrain = False
        
        thresholds = self.performance_data["thresholds"]
        overall_accuracy = self.performance_data["overall_stats"]["overall_accuracy"]
        
        # Check 1: Overall accuracy below minimum threshold
        if overall_accuracy < thresholds["min_accuracy"]:
            triggers.append(f"Overall accuracy ({overall_accuracy:.1f}%) below minimum threshold ({thresholds['min_accuracy']}%)")
            should_retrain = True
        
        # Check 2: Recent performance decline
        recent_7d = self.get_recent_performance(7)
        recent_30d = self.get_recent_performance(30)
        
        if (recent_7d["predictions"] >= thresholds["min_predictions_for_eval"] and 
            recent_30d["predictions"] >= thresholds["min_predictions_for_eval"]):
            
            accuracy_decline = recent_30d["accuracy"] - recent_7d["accuracy"]
            if accuracy_decline > thresholds["accuracy_decline_threshold"]:
                triggers.append(f"Recent accuracy decline: {accuracy_decline:.1f}% (7d: {recent_7d['accuracy']:.1f}% vs 30d: {recent_30d['accuracy']:.1f}%)")
                should_retrain = True
        
        # Check 3: Time since last training
        last_trained = datetime.fromisoformat(self.performance_data["model_info"]["last_trained"])
        days_since_training = (datetime.now() - last_trained).days
        
        if days_since_training > thresholds["days_without_retrain"]:
            triggers.append(f"Model hasn't been retrained for {days_since_training} days (threshold: {thresholds['days_without_retrain']} days)")
            should_retrain = True
        
        # Check 4: Low confidence predictions increasing
        if recent_7d["predictions"] >= 20:  # Minimum sample size
            if recent_7d["avg_confidence"] < 0.65:  # Low average confidence
                triggers.append(f"Recent predictions show low confidence (avg: {recent_7d['avg_confidence']:.1%})")
                # This is a warning, not necessarily requiring immediate retraining
        
        return should_retrain, triggers
    
    def get_retraining_recommendation(self) -> Dict:
        """Get comprehensive retraining recommendation"""
        should_retrain, triggers = self.check_retraining_triggers()
        recent_7d = self.get_recent_performance(7)
        recent_30d = self.get_recent_performance(30)
        
        # Determine urgency level
        urgency = "low"
        if should_retrain:
            if self.performance_data["overall_stats"]["overall_accuracy"] < 70:
                urgency = "high"
            elif len(triggers) >= 2:
                urgency = "medium"
        
        # Calculate estimated improvement potential
        improvement_potential = max(0, 85 - self.performance_data["overall_stats"]["overall_accuracy"])
        
        return {
            "should_retrain": should_retrain,
            "urgency": urgency,
            "triggers": triggers,
            "current_performance": {
                "overall_accuracy": self.performance_data["overall_stats"]["overall_accuracy"],
                "recent_7d_accuracy": recent_7d["accuracy"],
                "recent_30d_accuracy": recent_30d["accuracy"],
                "total_predictions": self.performance_data["overall_stats"]["total_predictions"]
            },
            "estimated_improvement": f"+{improvement_potential:.1f}%",
            "recommendation": self._generate_recommendation_text(should_retrain, urgency, triggers),
            "next_check": (datetime.now() + timedelta(days=1)).isoformat()
        }
    
    def _generate_recommendation_text(self, should_retrain: bool, urgency: str, triggers: List[str]) -> str:
        """Generate human-readable recommendation text"""
        if not should_retrain:
            return "✅ Model performance is stable. Continue monitoring."
        
        if urgency == "high":
            return "🚨 URGENT: Model performance has significantly declined. Immediate retraining recommended."
        elif urgency == "medium":
            return "⚠️ MODERATE: Multiple performance indicators suggest retraining within 2-3 days."
        else:
            return "📊 LOW: Consider retraining when convenient to maintain optimal performance."
    
    def simulate_prediction_results(self, num_days: int = 7):
        """Simulate some prediction results for demonstration"""
        import random
        
        for i in range(num_days):
            date = datetime.now() - timedelta(days=num_days - i - 1)
            
            # Simulate 3-8 predictions per day
            daily_predictions = random.randint(3, 8)
            
            for _ in range(daily_predictions):
                # Simulate accuracy decline over time (for demo)
                base_accuracy = 0.79 - (i * 0.01)  # Slight decline
                confidence = random.uniform(0.55, 0.95)
                
                # Higher confidence predictions are more likely to be correct
                prediction_correct = random.random() < (base_accuracy + (confidence - 0.7) * 0.2)
                
                self.add_prediction_result(prediction_correct, confidence)

# Example usage and testing
if __name__ == "__main__":
    # Create monitor instance
    monitor = ModelMonitor()
    
    # Simulate some recent predictions for demonstration
    print("Simulating recent prediction results...")
    monitor.simulate_prediction_results(14)
    
    # Get retraining recommendation
    recommendation = monitor.get_retraining_recommendation()
    
    print("\n" + "="*50)
    print("MODEL RETRAINING ANALYSIS")
    print("="*50)
    print(f"Should Retrain: {recommendation['should_retrain']}")
    print(f"Urgency Level: {recommendation['urgency'].upper()}")
    print(f"Recommendation: {recommendation['recommendation']}")
    
    if recommendation['triggers']:
        print(f"\nTriggers ({len(recommendation['triggers'])}):")
        for i, trigger in enumerate(recommendation['triggers'], 1):
            print(f"  {i}. {trigger}")
    
    print(f"\nCurrent Performance:")
    perf = recommendation['current_performance']
    print(f"  • Overall Accuracy: {perf['overall_accuracy']:.1f}%")
    print(f"  • Recent 7d Accuracy: {perf['recent_7d_accuracy']:.1f}%")
    print(f"  • Recent 30d Accuracy: {perf['recent_30d_accuracy']:.1f}%")
    print(f"  • Total Predictions: {perf['total_predictions']:,}")
    
    print(f"\nEstimated Improvement: {recommendation['estimated_improvement']}")
    print(f"Next Check: {recommendation['next_check'][:10]}")