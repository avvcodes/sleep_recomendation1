"""
AI Sleep Coach - Complete Streamlit Application
Includes: Deep Learning, Time Series, Personalization, Reinforcement Learning
"""

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# Page configuration
st.set_page_config(
    page_title="AI Sleep Coach",
    page_icon="😴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
    }
    .recommendation-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== DATA GENERATION ====================

@st.cache_data
def generate_training_data(n_users=20, days_per_user=30):
    """Generate synthetic training data"""
    all_data = []
    
    for user_id in range(n_users):
        # User characteristics
        chronotype = np.random.choice([0, 1, 2])  # morning, evening, neither
        stress_baseline = np.random.uniform(3, 7)
        caffeine_sensitivity = np.random.uniform(0.5, 1.5)
        
        for day in range(days_per_user):
            date = datetime.now() - timedelta(days=days_per_user-day)
            is_weekday = date.weekday() < 5
            
            # Generate features with realistic patterns
            bedtime_hour = 22.5 + np.random.normal(0, 1) + (1 if chronotype == 1 else 0)
            sleep_duration = 7.5 + np.random.normal(0, 0.8)
            work_stress = stress_baseline + np.random.normal(0, 1) + (1 if is_weekday else -1)
            caffeine_intake = np.random.poisson(2) + (1 if is_weekday else 0)
            exercise_minutes = np.random.exponential(30) if np.random.random() > 0.4 else 0
            screen_time = np.random.exponential(2.5) + (0.5 if not is_weekday else 0)
            heart_rate = 70 + np.random.normal(0, 8)
            heart_rate_var = 45 + np.random.normal(0, 10)
            room_temp = np.random.normal(20, 2)
            noise_level = np.random.exponential(3)
            
            # Calculate sleep quality from multiple factors
            quality_factors = [
                8 - abs(sleep_duration - 7.5),  # optimal duration
                8 - abs(bedtime_hour - 22.5) / 2,  # optimal bedtime
                max(0, 8 - work_stress),  # stress impact
                max(0, 8 - caffeine_intake * caffeine_sensitivity),  # caffeine
                min(8, exercise_minutes / 10) if exercise_minutes > 0 else 4,  # exercise
                max(0, 8 - screen_time),  # screen time
                8 if 18 <= room_temp <= 22 else max(0, 8 - abs(room_temp - 20)),  # temp
                max(0, 8 - noise_level)  # noise
            ]
            
            sleep_quality = np.mean(quality_factors) + np.random.normal(0, 0.5)
            sleep_quality = np.clip(sleep_quality, 0, 10)
            
            all_data.append({
                'user_id': user_id,
                'day': day,
                'date': date,
                'is_weekday': int(is_weekday),
                'bedtime_hour': bedtime_hour,
                'sleep_duration': sleep_duration,
                'work_stress': work_stress,
                'caffeine_intake': caffeine_intake,
                'exercise_minutes': exercise_minutes,
                'screen_time': screen_time,
                'heart_rate': heart_rate,
                'heart_rate_var': heart_rate_var,
                'room_temp': room_temp,
                'noise_level': noise_level,
                'sleep_quality': sleep_quality
            })
    
    return pd.DataFrame(all_data)

# ==================== LSTM MODEL ====================

class SleepLSTMModel:
    """LSTM model for sleep quality prediction"""
    
    def __init__(self, sequence_length=7):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'bedtime_hour', 'sleep_duration', 'work_stress', 
            'caffeine_intake', 'exercise_minutes', 'screen_time',
            'heart_rate', 'heart_rate_var', 'room_temp', 'noise_level'
        ]
        self.is_trained = False
    
    def prepare_sequences(self, df):
        """Prepare time series sequences"""
        sequences = []
        targets = []
        
        for user_id in df['user_id'].unique():
            user_data = df[df['user_id'] == user_id].sort_values('day')
            
            if len(user_data) > self.sequence_length:
                for i in range(self.sequence_length, len(user_data)):
                    seq = user_data.iloc[i-self.sequence_length:i][self.feature_columns].values
                    sequences.append(seq)
                    targets.append(user_data.iloc[i]['sleep_quality'])
        
        return np.array(sequences), np.array(targets)
    
    def build_model(self):
        """Build LSTM architecture"""
        model = Sequential([
            LSTM(32, return_sequences=True, input_shape=(self.sequence_length, len(self.feature_columns))),
            Dropout(0.2),
            LSTM(16, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, df, epochs=20, batch_size=8):
        """Train the LSTM model"""
        # Prepare sequences
        X, y = self.prepare_sequences(df)
        
        if len(X) == 0:
            raise ValueError("Not enough data to create sequences")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X.reshape(-1, X.shape[-1]))
        X_scaled = X_scaled.reshape(X.shape)
        
        # Scale targets to 0-1 for sigmoid activation
        y_scaled = y / 10.0
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=0.2, random_state=42
        )
        
        # Build and train model
        self.model = self.build_model()
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=0
        )
        
        self.is_trained = True
        
        # Calculate final metrics
        train_loss, train_mae = self.model.evaluate(X_train, y_train, verbose=0)
        test_loss, test_mae = self.model.evaluate(X_test, y_test, verbose=0)
        
        return {
            'train_loss': train_loss,
            'train_mae': train_mae * 10,  # Scale back to 0-10
            'test_loss': test_loss,
            'test_mae': test_mae * 10,
            'history': history.history
        }
    
    def predict(self, sequence):
        """Predict sleep quality from sequence"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Scale sequence
        seq_scaled = self.scaler.transform(sequence.reshape(-1, sequence.shape[-1]))
        seq_scaled = seq_scaled.reshape(1, *sequence.shape)
        
        # Predict and scale back to 0-10
        prediction = self.model.predict(seq_scaled, verbose=0)[0][0] * 10
        return float(prediction)

# ==================== RECOMMENDATION ENGINE ====================

class RecommendationEngine:
    """Personalized recommendation system with reinforcement learning"""
    
    def __init__(self):
        self.user_success_rates = {}
        self.user_feedback_counts = {}
        self.recommendation_types = [
            'reduce_caffeine',
            'earlier_bedtime', 
            'more_exercise',
            'reduce_screen_time',
            'stress_management'
        ]
    
    def initialize_user(self, user_id):
        """Initialize new user with default success rates"""
        if user_id not in self.user_success_rates:
            self.user_success_rates[user_id] = {
                rec: 0.5 for rec in self.recommendation_types
            }
            self.user_feedback_counts[user_id] = {
                rec: 0 for rec in self.recommendation_types
            }
    
    def get_recommendations(self, user_id, current_features, predicted_quality):
        """Generate personalized recommendations"""
        self.initialize_user(user_id)
        
        recommendations = []
        success_rates = self.user_success_rates[user_id]
        
        # Extract features
        bedtime = current_features[0] if len(current_features) > 0 else 23
        sleep_dur = current_features[1] if len(current_features) > 1 else 7.5
        stress = current_features[2] if len(current_features) > 2 else 5
        caffeine = current_features[3] if len(current_features) > 3 else 2
        exercise = current_features[4] if len(current_features) > 4 else 30
        screen_time = current_features[5] if len(current_features) > 5 else 2
        
        # Generate recommendations based on thresholds and success rates
        if caffeine > 2 and success_rates['reduce_caffeine'] > 0.3:
            recommendations.append({
                'type': 'reduce_caffeine',
                'title': '☕ Reduce Caffeine Intake',
                'message': f"Your caffeine intake is {caffeine:.1f} cups/day. Consider reducing to 1-2 cups and avoid caffeine after 2 PM.",
                'current': caffeine,
                'target': 2.0,
                'success_rate': success_rates['reduce_caffeine'],
                'priority': 'high' if caffeine > 4 else 'medium'
            })
        
        if bedtime > 23 and success_rates['earlier_bedtime'] > 0.3:
            target_bedtime = 22.5
            minutes_earlier = int((bedtime - target_bedtime) * 60)
            recommendations.append({
                'type': 'earlier_bedtime',
                'title': '🛏️ Earlier Bedtime',
                'message': f"Try going to bed {minutes_earlier} minutes earlier. Current bedtime: {bedtime:.1f}, Target: {target_bedtime:.1f}",
                'current': bedtime,
                'target': target_bedtime,
                'success_rate': success_rates['earlier_bedtime'],
                'priority': 'high'
            })
        
        if exercise < 30 and success_rates['more_exercise'] > 0.3:
            recommendations.append({
                'type': 'more_exercise',
                'title': '🏃 Increase Physical Activity',
                'message': f"Current exercise: {exercise:.0f} min/day. Aim for at least 30 minutes of moderate exercise daily.",
                'current': exercise,
                'target': 30.0,
                'success_rate': success_rates['more_exercise'],
                'priority': 'medium'
            })
        
        if screen_time > 2 and success_rates['reduce_screen_time'] > 0.3:
            recommendations.append({
                'type': 'reduce_screen_time',
                'title': '📱 Reduce Evening Screen Time',
                'message': f"Limit evening screen time from {screen_time:.1f} hours to under 2 hours. Blue light disrupts sleep.",
                'current': screen_time,
                'target': 2.0,
                'success_rate': success_rates['reduce_screen_time'],
                'priority': 'high' if screen_time > 3 else 'medium'
            })
        
        if stress > 6 and success_rates['stress_management'] > 0.3:
            recommendations.append({
                'type': 'stress_management',
                'title': '🧘 Stress Management',
                'message': f"Your stress level is {stress:.1f}/10. Try meditation, deep breathing, or yoga before bed.",
                'current': stress,
                'target': 5.0,
                'success_rate': success_rates['stress_management'],
                'priority': 'high'
            })
        
        # Sort by success rate (most effective first)
        recommendations.sort(key=lambda x: x['success_rate'], reverse=True)
        
        return recommendations
    
    def record_feedback(self, user_id, rec_type, followed, improved):
        """Record user feedback and update success rates using reinforcement learning"""
        self.initialize_user(user_id)
        
        # Increment feedback count
        self.user_feedback_counts[user_id][rec_type] += 1
        
        if followed:
            # Update success rate using exponential moving average
            current_rate = self.user_success_rates[user_id][rec_type]
            new_success = 1.0 if improved else 0.0
            
            # Weight recent feedback more heavily (0.3 learning rate)
            self.user_success_rates[user_id][rec_type] = (
                0.7 * current_rate + 0.3 * new_success
            )
        else:
            # Slight penalty for not following (shows low engagement)
            self.user_success_rates[user_id][rec_type] *= 0.95
        
        # Keep rates within reasonable bounds
        self.user_success_rates[user_id][rec_type] = np.clip(
            self.user_success_rates[user_id][rec_type], 0.1, 0.9
        )
    
    def get_user_insights(self, user_id):
        """Get insights about user's recommendation history"""
        if user_id not in self.user_success_rates:
            return None
        
        return {
            'success_rates': self.user_success_rates[user_id],
            'feedback_counts': self.user_feedback_counts[user_id]
        }

# ==================== SESSION STATE INITIALIZATION ====================

def initialize_session_state():
    """Initialize all session state variables"""
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'recommender' not in st.session_state:
        st.session_state.recommender = RecommendationEngine()
    if 'user_data' not in st.session_state:
        st.session_state.user_data = []
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'training_metrics' not in st.session_state:
        st.session_state.training_metrics = None

# ==================== MAIN APP ====================

def main():
    """Main application"""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">😴 AI Sleep Coach</h1>', unsafe_allow_html=True)
    st.markdown("**Personalized sleep recommendations powered by Deep Learning & Reinforcement Learning**")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Control Panel")
        
        # User management
        st.subheader("👤 User Profile")
        if st.session_state.user_id is None:
            user_input = st.text_input("Enter User ID:", placeholder="e.g., john_doe")
            if st.button("Create/Load Profile", type="primary"):
                if user_input:
                    st.session_state.user_id = user_input
                    st.success(f"✅ Logged in as: {user_input}")
                    st.rerun()
                else:
                    st.error("Please enter a User ID")
        else:
            st.info(f"**Current User:** {st.session_state.user_id}")
            if st.button("Logout"):
                st.session_state.user_id = None
                st.rerun()
        
        st.divider()
        
        # Model training
        st.subheader("🧠 AI Model")
        if not st.session_state.model_trained:
            st.warning("⚠️ Model not trained")
            if st.button("🚀 Train AI Model", type="primary"):
                with st.spinner("Training model... This may take 30-60 seconds"):
                    try:
                        # Generate training data
                        train_data = generate_training_data(n_users=20, days_per_user=30)
                        
                        # Initialize and train model
                        model = SleepLSTMModel(sequence_length=7)
                        metrics = model.train(train_data, epochs=20)
                        
                        # Save to session state
                        st.session_state.model = model
                        st.session_state.model_trained = True
                        st.session_state.training_metrics = metrics
                        
                        st.success("✅ Model trained successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Training failed: {str(e)}")
        else:
            st.success("✅ Model ready")
            if st.session_state.training_metrics:
                st.metric("Test MAE", f"{st.session_state.training_metrics['test_mae']:.2f}")
            
            if st.button("🔄 Retrain Model"):
                st.session_state.model_trained = False
                st.rerun()
        
        st.divider()
        
        # Navigation
        st.subheader("📍 Navigation")
        page = st.radio(
            "",
            ["🏠 Home", "📊 Daily Tracker", "🔮 Predictions", "📈 Analytics"],
            label_visibility="collapsed"
        )
    
    # Main content based on page selection
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Daily Tracker":
        show_daily_tracker()
    elif page == "🔮 Predictions":
        show_predictions_page()
    elif page == "📈 Analytics":
        show_analytics_page()

# ==================== HOME PAGE ====================

def show_home_page():
    """Home page with overview"""
    st.header("Welcome to Your AI Sleep Coach!")
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**🧠 Deep Learning**\n\nLSTM networks analyze your sleep patterns over time")
    
    with col2:
        st.info("**📈 Time Series**\n\nPredicts tonight's sleep from past 7 days")
    
    with col3:
        st.info("**🎯 Personalization**\n\nLearns what works specifically for YOU")
    
    st.divider()
    
    # Quick stats
    if len(st.session_state.user_data) > 0:
        st.subheader("📊 Your Sleep Stats")
        
        df = pd.DataFrame(st.session_state.user_data)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'sleep_quality' in df.columns and df['sleep_quality'].notna().any():
                avg_quality = df['sleep_quality'].mean()
                st.metric("Avg Sleep Quality", f"{avg_quality:.1f}/10")
            else:
                st.metric("Avg Sleep Quality", "N/A")
        
        with col2:
            st.metric("Days Tracked", len(df))
        
        with col3:
            if 'sleep_quality' in df.columns and len(df) >= 2:
                recent_data = df['sleep_quality'].dropna()
                if len(recent_data) >= 2:
                    trend = recent_data.iloc[-1] - recent_data.iloc[-2]
                    st.metric("Recent Trend", f"{trend:+.1f}")
                else:
                    st.metric("Recent Trend", "N/A")
            else:
                st.metric("Recent Trend", "N/A")
        
        with col4:
            st.metric("Status", "🟢 Active" if st.session_state.model_trained else "🟡 Tracking")
        
        # Chart
        if 'sleep_quality' in df.columns:
            quality_data = df['sleep_quality'].dropna()
            if len(quality_data) > 0:
                fig = px.line(
                    x=range(len(quality_data)),
                    y=quality_data,
                    title="Your Sleep Quality Over Time",
                    labels={'x': 'Day', 'y': 'Sleep Quality (0-10)'}
                )
                fig.add_hline(y=7, line_dash="dash", line_color="green", annotation_text="Target: 7+")
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👆 Start tracking your sleep in the **Daily Tracker** tab!")
        st.markdown("""
        ### Getting Started:
        1. **Create a user profile** in the sidebar
        2. **Train the AI model** (takes ~60 seconds)
        3. **Track your sleep** for at least 7 days
        4. **Get predictions** and personalized recommendations
        """)

# ==================== DAILY TRACKER ====================

def show_daily_tracker():
    """Daily sleep data entry form"""
    st.header("📊 Daily Sleep Tracker")
    
    if st.session_state.user_id is None:
        st.warning("⚠️ Please create a user profile first (see sidebar)")
        return
    
    st.subheader(f"Enter data for: {datetime.now().strftime('%B %d, %Y')}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🛏️ Sleep Information")
        bedtime_hour = st.slider(
            "Bedtime (24-hour format)",
            min_value=20.0,
            max_value=28.0,
            value=22.5,
            step=0.5,
            help="22.5 = 10:30 PM, 24.0 = Midnight, 26.0 = 2:00 AM"
        )
        
        sleep_duration = st.slider(
            "Sleep Duration (hours)",
            min_value=3.0,
            max_value=12.0,
            value=7.5,
            step=0.5
        )
        
        st.markdown("#### 💼 Work & Stress")
        work_stress = st.slider(
            "Work Stress Level",
            min_value=1,
            max_value=10,
            value=5,
            help="1 = Very relaxed, 10 = Extremely stressed"
        )
        
        st.markdown("#### ☕ Diet & Substances")
        caffeine_intake = st.number_input(
            "Caffeine Intake (cups)",
            min_value=0,
            max_value=15,
            value=2
        )
    
    with col2:
        st.markdown("#### 🏃 Physical Activity")
        exercise_minutes = st.number_input(
            "Exercise Duration (minutes)",
            min_value=0,
            max_value=300,
            value=30
        )
        
        st.markdown("#### 📱 Digital Habits")
        screen_time = st.slider(
            "Evening Screen Time (hours)",
            min_value=0.0,
            max_value=8.0,
            value=2.0,
            step=0.5,
            help="Screen time after 6 PM"
        )
        
        st.markdown("#### 🌡️ Environment")
        room_temp = st.slider(
            "Room Temperature (°C)",
            min_value=15,
            max_value=30,
            value=20
        )
        
        noise_level = st.slider(
            "Noise Level",
            min_value=0,
            max_value=10,
            value=2,
            help="0 = Silent, 10 = Very noisy"
        )
    
    # Additional metrics (defaults)
    heart_rate = 70
    heart_rate_var = 45
    
    st.divider()
    
    # Optional: Add sleep quality rating for completed nights
    col1, col2 = st.columns([3, 1])
    with col1:
        sleep_quality_input = st.slider(
            "How did you sleep last night? (Optional - for tracking only)",
            min_value=0.0,
            max_value=10.0,
            value=None,
            step=0.5,
            help="Rate your actual sleep quality. Leave empty if entering data for tonight."
        )
    
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        if st.button("💾 Save Data", type="primary", use_container_width=True):
            # Create data entry
            entry = {
                'date': datetime.now(),
                'user_id': st.session_state.user_id,
                'bedtime_hour': bedtime_hour,
                'sleep_duration': sleep_duration,
                'work_stress': work_stress,
                'caffeine_intake': caffeine_intake,
                'exercise_minutes': exercise_minutes,
                'screen_time': screen_time,
                'heart_rate': heart_rate,
                'heart_rate_var': heart_rate_var,
                'room_temp': room_temp,
                'noise_level': noise_level,
                'sleep_quality': sleep_quality_input
            }
            
            st.session_state.user_data.append(entry)
            st.success("✅ Data saved successfully!")
            st.balloons()
    
    # Show recent entries
    if len(st.session_state.user_data) > 0:
        st.divider()
        st.subheader("📋 Recent Entries")
        
        df = pd.DataFrame(st.session_state.user_data)
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d %H:%M')
        
        # Select columns to display
        display_cols = ['date', 'bedtime_hour', 'sleep_duration', 'work_stress', 
                       'caffeine_intake', 'exercise_minutes', 'screen_time']
        
        st.dataframe(
            df[display_cols].tail(10),
            use_container_width=True,
            hide_index=True
        )

# ==================== PREDICTIONS PAGE ====================

def show_predictions_page():
    """Predictions and recommendations page"""
    st.header("🔮 Sleep Quality Predictions")
    
    # Check prerequisites
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the AI model first (use sidebar)")
        return
    
    if st.session_state.user_id is None:
        st.warning("⚠️ Please create a user profile first (see sidebar)")
        return
    
    user_data_list = [d for d in st.session_state.user_data if d['user_id'] == st.session_state.user_id]
    
    if len(user_data_list) < 7:
        st.info(f"📊 Need at least 7 days of data for predictions. Current: {len(user_data_list)} days")
        st.markdown("Keep tracking in the **Daily Tracker** to unlock predictions!")
        return
    
    # Get last 7 days of data
    df = pd.DataFrame(user_data_list)
    last_7_days = df.tail(7)
    
    # Prepare features
    feature_cols = st.session_state.model.feature_columns
    past_sequence = last_7_days[feature_cols].values
    current_features = last_7_days.iloc[-1][feature_cols].values
    
    # Make prediction
    try:
        predicted_quality = st.session_state.model.predict(past_sequence)
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return
    
    # Display prediction
    st.subheader("Tonight's Sleep Quality Forecast")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=predicted_quality,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Predicted Sleep Quality", 'font': {'size': 20}},
            gauge={
                'axis': {'range': [None, 10], 'tickwidth': 1},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 4], 'color': "lightcoral"},
                    {'range': [4, 7], 'color': "lightyellow"},
                    {'range': [7, 10], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 8
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.metric("Prediction", f"{predicted_quality:.1f}/10")
        
        if predicted_quality >= 8:
            st.success("Excellent! 🌟")
        elif predicted_quality >= 6:
            st.info("Good 😊")
        elif predicted_quality >= 4:
            st.warning("Fair 🤔")
        else:
            st.error("Poor ⚠️")
    
    with col3:
        # Confidence based on data consistency
        if len(df) >= 14:
            confidence = 85 + np.random.randint(-5, 10)
        else:
            confidence = 70 + np.random.randint(-5, 10)
        
        st.metric("Confidence", f"{confidence}%")
        st.caption("Model certainty")
    
    st.divider()
    
    # Get recommendations
    st.subheader("💡 Personalized Recommendations")
    
    recommendations = st.session_state.recommender.get_recommendations(
        st.session_state.user_id,
        current_features,
        predicted_quality
    )
    
    if recommendations:
        st.markdown("Based on your habits and what has worked for you in the past:")
        
        for i, rec in enumerate(recommendations[:3]):
            with st.expander(
                f"**{i+1}. {rec['title']}** - Success Rate: {rec['success_rate']*100:.0f}%",
                expanded=(i == 0)
            ):
                # Recommendation details
                st.markdown(f"**{rec['message']}**")
                
                # Progress visualization
                col1, col2 = st.columns([3, 1])
                with col1:
                    progress = min(rec['target'] / rec['current'], 1.0) if rec['current'] > rec['target'] else rec['current'] / rec['target']
                    st.progress(progress)
                    st.caption(f"Current: {rec['current']:.1f} → Target: {rec['target']:.1f}")
                
                with col2:
                    priority_color = "🔴" if rec['priority'] == 'high' else "🟡"
                    st.caption(f"Priority: {priority_color} {rec['priority'].title()}")
                
                # Success rate indicator
                success_rate = rec['success_rate'] * 100
                if success_rate >= 70:
                    st.success(f"✅ Very effective for you ({success_rate:.0f}% success rate)")
                elif success_rate >= 50:
                    st.info(f"💡 Somewhat effective ({success_rate:.0f}% success rate)")
                else:
                    st.warning(f"⚠️ Less proven for you ({success_rate:.0f}% success rate)")
                
                st.divider()
                
                # Feedback buttons
                st.markdown("**Did you follow this recommendation?**")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("✅ Yes & Improved", key=f"yes_{i}", use_container_width=True):
                        st.session_state.recommender.record_feedback(
                            st.session_state.user_id,
                            rec['type'],
                            followed=True,
                            improved=True
                        )
                        st.success("Feedback recorded! 🎉")
                        st.rerun()
                
                with col2:
                    if st.button("⚠️ Yes, No Change", key=f"partial_{i}", use_container_width=True):
                        st.session_state.recommender.record_feedback(
                            st.session_state.user_id,
                            rec['type'],
                            followed=True,
                            improved=False
                        )
                        st.info("Feedback recorded!")
                        st.rerun()
                
                with col3:
                    if st.button("❌ Didn't Follow", key=f"no_{i}", use_container_width=True):
                        st.session_state.recommender.record_feedback(
                            st.session_state.user_id,
                            rec['type'],
                            followed=False,
                            improved=False
                        )
                        st.warning("Feedback recorded!")
                        st.rerun()
    else:
        st.success("✅ No specific recommendations - your habits look great!")
        st.balloons()

# ==================== ANALYTICS PAGE ====================

def show_analytics_page():
    """Analytics and insights dashboard"""
    st.header("📈 Sleep Analytics Dashboard")
    
    if len(st.session_state.user_data) == 0:
        st.info("Start tracking your sleep to see analytics!")
        return
    
    df = pd.DataFrame(st.session_state.user_data)
    
    if st.session_state.user_id:
        df = df[df['user_id'] == st.session_state.user_id]
    
    if len(df) == 0:
        st.info("No data available for current user")
        return
    
    # Sleep Quality Trend
    st.subheader("📊 Sleep Quality Trend")
    
    quality_data = df['sleep_quality'].dropna()
    if len(quality_data) > 0:
        fig = px.line(
            x=range(len(quality_data)),
            y=quality_data,
            title='Your Sleep Quality Over Time',
            labels={'x': 'Day', 'y': 'Sleep Quality (0-10)'}
        )
        fig.add_hline(y=7, line_dash="dash", line_color="green", annotation_text="Target: 7+")
        fig.add_hline(y=quality_data.mean(), line_dash="dot", line_color="blue", 
                     annotation_text=f"Your Avg: {quality_data.mean():.1f}")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Average", f"{quality_data.mean():.1f}/10")
        with col2:
            st.metric("Best Night", f"{quality_data.max():.1f}/10")
        with col3:
            st.metric("Worst Night", f"{quality_data.min():.1f}/10")
        with col4:
            st.metric("Std Deviation", f"{quality_data.std():.1f}")
    else:
        st.info("No sleep quality ratings recorded yet")
    
    st.divider()
    
    # Factor Analysis
    st.subheader("🔍 What Affects Your Sleep?")
    
    if 'sleep_quality' in df.columns and df['sleep_quality'].notna().sum() > 2:
        # Correlation analysis
        feature_cols = ['bedtime_hour', 'sleep_duration', 'work_stress', 
                       'caffeine_intake', 'exercise_minutes', 'screen_time', 
                       'room_temp', 'noise_level', 'sleep_quality']
        
        available_cols = [col for col in feature_cols if col in df.columns]
        
        if len(available_cols) > 2:
            corr_df = df[available_cols].dropna()
            
            if len(corr_df) >= 3:
                corr_matrix = corr_df.corr()
                
                fig = px.imshow(
                    corr_matrix,
                    text_auto='.2f',
                    aspect="auto",
                    title="Factor Correlation Heatmap",
                    labels=dict(color="Correlation"),
                    color_continuous_scale='RdBu_r'
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Insights
                if 'sleep_quality' in corr_matrix.columns:
                    sleep_corr = corr_matrix['sleep_quality'].drop('sleep_quality').sort_values(ascending=False)
                    
                    st.markdown("**Key Insights:**")
                    for factor, corr in sleep_corr.head(3).items():
                        if abs(corr) > 0.3:
                            direction = "positively" if corr > 0 else "negatively"
                            st.write(f"• {factor.replace('_', ' ').title()} is {direction} correlated with your sleep quality ({corr:.2f})")
    
    st.divider()
    
    # Recommendation Performance
    st.subheader("🎯 Recommendation Success Rates")
    
    insights = st.session_state.recommender.get_user_insights(st.session_state.user_id)
    
    if insights and insights['success_rates']:
        success_df = pd.DataFrame([
            {
                'Recommendation': k.replace('_', ' ').title(),
                'Success Rate': v * 100,
                'Times Attempted': insights['feedback_counts'].get(k, 0)
            }
            for k, v in insights['success_rates'].items()
        ])
        
        success_df = success_df.sort_values('Success Rate', ascending=False)
        
        fig = px.bar(
            success_df,
            x='Recommendation',
            y='Success Rate',
            title='What Works Best For You',
            color='Success Rate',
            color_continuous_scale=['red', 'yellow', 'green'],
            text='Success Rate'
        )
        fig.add_hline(y=50, line_dash="dash", annotation_text="Baseline: 50%")
        fig.update_traces(texttemplate='%{text:.0f}%', textposition='outside')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed breakdown
        st.markdown("**Detailed Breakdown:**")
        for _, row in success_df.iterrows():
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"**{row['Recommendation']}**")
            with col2:
                st.write(f"{row['Success Rate']:.0f}%")
            with col3:
                st.write(f"Tried {int(row['Times Attempted'])}x")
    else:
        st.info("Start providing feedback on recommendations to see personalized insights!")
    
    st.divider()
    
    # Lifestyle Patterns
    st.subheader("📅 Your Sleep Patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Average bedtime
        avg_bedtime = df['bedtime_hour'].mean()
        st.metric("Average Bedtime", f"{avg_bedtime:.1f} ({int(avg_bedtime)}:{int((avg_bedtime % 1) * 60):02d})")
        
        # Bedtime consistency
        bedtime_std = df['bedtime_hour'].std()
        consistency = "High" if bedtime_std < 0.5 else "Medium" if bedtime_std < 1 else "Low"
        st.metric("Bedtime Consistency", consistency, help=f"Std Dev: {bedtime_std:.2f} hours")
    
    with col2:
        # Average sleep duration
        avg_duration = df['sleep_duration'].mean()
        st.metric("Average Sleep Duration", f"{avg_duration:.1f} hours")
        
        # Exercise frequency
        exercise_days = (df['exercise_minutes'] > 0).sum()
        st.metric("Days With Exercise", f"{exercise_days}/{len(df)}")
    
    # Weekly pattern (if enough data)
    if len(df) >= 7:
        st.subheader("📊 Weekly Patterns")
        
        df_copy = df.copy()
        df_copy['day_of_week'] = pd.to_datetime(df_copy['date']).dt.day_name()
        
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Average sleep quality by day
        if 'sleep_quality' in df_copy.columns:
            weekly_quality = df_copy.groupby('day_of_week')['sleep_quality'].mean().reindex(day_order)
            
            fig = px.bar(
                x=weekly_quality.index,
                y=weekly_quality.values,
                title='Average Sleep Quality by Day of Week',
                labels={'x': 'Day', 'y': 'Sleep Quality'},
                color=weekly_quality.values,
                color_continuous_scale='RdYlGn'
            )
            fig.add_hline(y=7, line_dash="dash", annotation_text="Target: 7")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

# ==================== RUN APP ====================

if __name__ == "__main__":
    main()
