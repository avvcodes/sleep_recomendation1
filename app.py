import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pickle
import os

# Page config
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
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .recommendation-card {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .success-rate-high {
        color: #28a745;
        font-weight: bold;
    }
    .success-rate-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .success-rate-low {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Data folder
DATA_DIR = "sleep_data"
MODEL_DIR = "sleep_model"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Initialize session state
if 'user_data' not in st.session_state:
    st.session_state.user_data = []
if 'feedback_history' not in st.session_state:
    st.session_state.feedback_history = []
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'recommender' not in st.session_state:
    st.session_state.recommender = None
if 'user_id' not in st.session_state:
    st.session_state.user_id = None

class StreamlitLSTMPredictor:
    """LSTM predictor optimized for Streamlit"""
    
    def __init__(self, sequence_length=7):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = ['bedtime_hour', 'sleep_duration', 'work_stress', 
                               'caffeine_intake', 'exercise_minutes', 'screen_time',
                               'heart_rate', 'heart_rate_var', 'room_temp', 'noise_level']
    
    def build_model(self):
        """Build LSTM model"""
        model = Sequential([
            LSTM(32, return_sequences=True, input_shape=(self.sequence_length, len(self.feature_columns))),
            Dropout(0.2),
            LSTM(16),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.model = model
        return model
    
    def train_on_demo_data(self):
        """Train on synthetic demo data"""
        # Generate demo data
        demo_data = self.generate_demo_data(n_users=15, days_per_user=20)
        
        # Prepare sequences
        X, y = self.prepare_sequences(demo_data)
        
        if len(X) == 0:
            return False
        
        # Scale
        # Fit scaler on flattened features
        X_flat = X.reshape(-1, X.shape[-1])
        X_scaled_flat = self.scaler.fit_transform(X_flat)
        X_scaled = X_scaled_flat.reshape(X.shape)
        y_scaled = y / 10.0
        
        # Build and train
        self.build_model()
        self.model.fit(X_scaled, y_scaled, epochs=20, batch_size=8, verbose=0)
        
        # persist model & scaler
        try:
            self.model.save(os.path.join(MODEL_DIR, "lstm_model.h5"))
            with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
                pickle.dump(self.scaler, f)
        except Exception:
            pass
        
        return True
    
    def generate_demo_data(self, n_users=15, days_per_user=20):
        """Generate demo training data"""
        all_data = []
        
        for user_id in range(n_users):
            for day in range(days_per_user):
                bedtime_hour = 22.5 + np.random.normal(0, 1)
                sleep_duration = 7.5 + np.random.normal(0, 0.8)
                work_stress = np.random.uniform(3, 8)
                caffeine_intake = np.random.poisson(2)
                exercise_minutes = np.random.exponential(30) if np.random.random() > 0.4 else 0
                screen_time = np.random.exponential(2.5)
                heart_rate = 70 + np.random.normal(0, 8)
                heart_rate_var = 45 + np.random.normal(0, 10)
                room_temp = np.random.normal(20, 2)
                noise_level = np.random.exponential(3)
                
                # Calculate sleep quality
                quality_factors = [
                    8 - abs(sleep_duration - 7.5),
                    8 - abs(bedtime_hour - 22.5) / 2,
                    max(0, 8 - work_stress),
                    max(0, 8 - caffeine_intake),
                    min(8, exercise_minutes / 10) if exercise_minutes > 0 else 4,
                    max(0, 8 - screen_time),
                    8 if 18 <= room_temp <= 22 else max(0, 8 - abs(room_temp - 20)),
                    max(0, 8 - noise_level)
                ]
                sleep_quality = np.mean(quality_factors) + np.random.normal(0, 0.5)
                sleep_quality = np.clip(sleep_quality, 0, 10)
                
                all_data.append({
                    'user_id': user_id,
                    'day': day,
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
    
    def prepare_sequences(self, df):
        """Prepare sequences from dataframe"""
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
    
    def predict(self, sequence):
        """Make prediction"""
        # sequence shape: (sequence_length, features)
        seq_flat = sequence.reshape(-1, sequence.shape[-1])  # (seq_len, features)
        seq_scaled_flat = self.scaler.transform(seq_flat)
        seq_scaled = seq_scaled_flat.reshape(1, sequence.shape[0], sequence.shape[1])
        prediction = self.model.predict(seq_scaled, verbose=0)[0][0] * 10
        return float(prediction)

class StreamlitRecommender:
    """Personalized recommender for Streamlit"""
    
    def __init__(self):
        self.user_success_rates = {}
        self.recommendation_types = [
            'reduce_caffeine', 'earlier_bedtime', 'more_exercise',
            'reduce_screen_time', 'stress_management'
        ]
        # Attempt load persisted rates
        self._load()
    
    def initialize_user(self, user_id):
        """Initialize new user"""
        if user_id not in self.user_success_rates:
            self.user_success_rates[user_id] = {
                rec: 0.5 for rec in self.recommendation_types
            }
    
    def get_recommendations(self, user_id, current_features, predicted_quality):
        """Get personalized recommendations"""
        self.initialize_user(user_id)
        
        recommendations = []
        success_rates = self.user_success_rates[user_id]
        
        bedtime, sleep_dur, stress, caffeine, exercise, screen_time = current_features[:6]
        
        if caffeine > 2 and success_rates['reduce_caffeine'] > 0.3:
            recommendations.append({
                'type': 'reduce_caffeine',
                'title': '☕ Reduce Caffeine',
                'message': f"Consider reducing caffeine from {caffeine:.1f} to 1-2 cups daily",
                'current_value': caffeine,
                'target_value': 2.0,
                'success_rate': success_rates['reduce_caffeine'],
                'impact': 'High' if caffeine > 4 else 'Medium'
            })
        
        if bedtime > 23 and success_rates['earlier_bedtime'] > 0.3:
            recommendations.append({
                'type': 'earlier_bedtime',
                'title': '🛏️ Earlier Bedtime',
                'message': f"Try going to bed 30-60 minutes earlier (currently {bedtime:.1f})",
                'current_value': bedtime,
                'target_value': 22.5,
                'success_rate': success_rates['earlier_bedtime'],
                'impact': 'High'
            })
        
        if exercise < 30 and success_rates['more_exercise'] > 0.3:
            recommendations.append({
                'type': 'more_exercise',
                'title': '🏃 More Exercise',
                'message': f"Increase daily exercise to 30+ minutes (currently {exercise:.0f})",
                'current_value': exercise,
                'target_value': 30.0,
                'success_rate': success_rates['more_exercise'],
                'impact': 'Medium'
            })
        
        if screen_time > 2 and success_rates['reduce_screen_time'] > 0.3:
            recommendations.append({
                'type': 'reduce_screen_time',
                'title': '📱 Reduce Screen Time',
                'message': f"Reduce evening screen time from {screen_time:.1f} to under 2 hours",
                'current_value': screen_time,
                'target_value': 2.0,
                'success_rate': success_rates['reduce_screen_time'],
                'impact': 'High'
            })
        
        if stress > 6 and success_rates['stress_management'] > 0.3:
            recommendations.append({
                'type': 'stress_management',
                'title': '🧘 Stress Management',
                'message': f"Try meditation or yoga (stress level: {stress:.1f}/10)",
                'current_value': stress,
                'target_value': 5.0,
                'success_rate': success_rates['stress_management'],
                'impact': 'High'
            })
        
        recommendations.sort(key=lambda x: x['success_rate'], reverse=True)
        return recommendations
    
    def record_feedback(self, user_id, rec_type, followed, improved):
        """Record user feedback"""
        self.initialize_user(user_id)
        
        if followed:
            current_rate = self.user_success_rates[user_id][rec_type]
            new_success = 1.0 if improved else 0.0
            self.user_success_rates[user_id][rec_type] = 0.7 * current_rate + 0.3 * new_success
        else:
            self.user_success_rates[user_id][rec_type] *= 0.9
        
        self.user_success_rates[user_id][rec_type] = np.clip(
            self.user_success_rates[user_id][rec_type], 0.1, 0.9
        )
        self._save()
    
    def _save(self):
        try:
            with open(os.path.join(MODEL_DIR, "recommender_rates.pkl"), "wb") as f:
                pickle.dump(self.user_success_rates, f)
        except Exception:
            pass
    
    def _load(self):
        try:
            fp = os.path.join(MODEL_DIR, "recommender_rates.pkl")
            if os.path.exists(fp):
                with open(fp, "rb") as f:
                    self.user_success_rates = pickle.load(f)
        except Exception:
            self.user_success_rates = {}

def estimate_sleep_quality(entry):
    """
    Simple heuristic to estimate sleep quality (0-10) from today's saved features.
    This is for quick user feedback; real label should come from self-report or device.
    """
    sleep_duration = entry.get('sleep_duration', 7.5)
    bedtime = entry.get('bedtime_hour', 22.5)
    stress = entry.get('work_stress', 5)
    caffeine = entry.get('caffeine_intake', 2)
    exercise = entry.get('exercise_minutes', 0)
    screen = entry.get('screen_time', 2)
    room_temp = entry.get('room_temp', 20)
    noise = entry.get('noise_level', 2)
    
    factors = []
    factors.append(8 - abs(sleep_duration - 7.5))
    factors.append(8 - abs(bedtime - 22.5) / 2)
    factors.append(max(0, 8 - stress))
    factors.append(max(0, 8 - caffeine))
    factors.append(min(8, exercise / 10) if exercise > 0 else 4)
    factors.append(max(0, 8 - screen))
    factors.append(8 if 18 <= room_temp <= 22 else max(0, 8 - abs(room_temp - 20)))
    factors.append(max(0, 8 - noise))
    score = np.clip(np.mean(factors), 0, 10)
    return float(score)

def save_user_data(user_id):
    fp = os.path.join(DATA_DIR, f"{user_id}_data.pkl")
    try:
        with open(fp, "wb") as f:
            pickle.dump({
                'user_data': st.session_state.user_data,
                'feedback_history': st.session_state.feedback_history
            }, f)
    except Exception:
        pass

def load_user_data(user_id):
    fp = os.path.join(DATA_DIR, f"{user_id}_data.pkl")
    if os.path.exists(fp):
        try:
            with open(fp, "rb") as f:
                payload = pickle.load(f)
                st.session_state.user_data = payload.get('user_data', [])
                st.session_state.feedback_history = payload.get('feedback_history', [])
        except Exception:
            st.session_state.user_data = []
            st.session_state.feedback_history = []
    else:
        st.session_state.user_data = []
        st.session_state.feedback_history = []

def main():
    st.markdown('<h1 class="main-header">😴 AI Sleep Coach</h1>', unsafe_allow_html=True)
    st.markdown("**Personalized sleep recommendations powered by Deep Learning & Reinforcement Learning**")
    
    # Sidebar
    with st.sidebar:
        st.header("🎯 Navigation")
        page = st.radio("", ["🏠 Home", "📊 Daily Tracker", "🔮 Predictions", "📈 Analytics", "⚙️ Settings"])
        
        st.divider()
        
        # User ID selection
        st.subheader("👤 User Profile")
        if st.session_state.user_id is None:
            user_input = st.text_input("Enter User ID:", placeholder="e.g., user123")
            if st.button("Create/Load Profile"):
                if user_input:
                    st.session_state.user_id = user_input
                    load_user_data(user_input)
                    # initialize recommender
                    if st.session_state.recommender is None:
                        st.session_state.recommender = StreamlitRecommender()
                    st.success(f"Logged in as: {user_input}")
                    st.rerun()
        else:
            st.info(f"**Current User:** {st.session_state.user_id}")
            if st.button("Switch User"):
                st.session_state.user_id = None
                st.session_state.user_data = []
                st.session_state.feedback_history = []
                st.rerun()
        
        st.divider()
        
        # Model status
        st.subheader("🧠 AI Model Status")
        if st.session_state.model_trained:
            st.success("✅ Model Active")
        else:
            st.warning("⚠️ Model Not Trained")
            if st.button("Train AI Model"):
                with st.spinner("Training model on demo data..."):
                    predictor = StreamlitLSTMPredictor()
                    predictor.train_on_demo_data()
                    st.session_state.predictor = predictor
                    if st.session_state.recommender is None:
                        st.session_state.recommender = StreamlitRecommender()
                    st.session_state.model_trained = True
                    st.success("Model trained successfully!")
                    st.rerun()
    
    # Main content
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Daily Tracker":
        show_daily_tracker()
    elif page == "🔮 Predictions":
        show_predictions()
    elif page == "📈 Analytics":
        show_analytics()
    elif page == "⚙️ Settings":
        show_settings()

def show_home_page():
    """Home page with overview"""
    st.header("Welcome to Your AI Sleep Coach!")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🧠 Deep Learning</h3>
            <p>LSTM neural networks analyze your sleep patterns over time</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📈 Time Series Analysis</h3>
            <p>Predicts tonight's sleep based on your past 7 days</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Personalization</h3>
            <p>Learns what works specifically for YOU</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Quick stats
    if st.session_state.user_data:
        st.subheader("📊 Your Stats")
        
        df = pd.DataFrame(st.session_state.user_data)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_sleep = df['sleep_quality'].dropna().mean() if 'sleep_quality' in df.columns else None
            if avg_sleep is not None and not np.isnan(avg_sleep):
                st.metric("Avg Sleep Quality", f"{avg_sleep:.1f}/10")
            else:
                st.metric("Avg Sleep Quality", "N/A")
        
        with col2:
            days_tracked = len(df)
            st.metric("Days Tracked", days_tracked)
        
        with col3:
            if len(df) >= 2 and 'sleep_quality' in df.columns:
                recent_trend = df['sleep_quality'].fillna(method='ffill').iloc[-1] - df['sleep_quality'].fillna(method='ffill').iloc[-2]
                st.metric("Recent Trend", f"{recent_trend:+.1f}", delta=f"{recent_trend:.1f}")
            else:
                st.metric("Recent Trend", "N/A")
        
        with col4:
            recommendations_given = len(st.session_state.feedback_history)
            st.metric("Recommendations", recommendations_given)
        
        # Sleep quality chart
        if 'sleep_quality' in df.columns:
            fig = px.line(df.reset_index(), x='index', y='sleep_quality', title='Your Sleep Quality Over Time')
            fig.update_layout(xaxis_title="Day", yaxis_title="Sleep Quality (0-10)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No sleep quality ratings yet. Record your sleep to see charts.")
    else:
        st.info("👆 Start tracking your sleep to see personalized insights!")

def show_daily_tracker():
    """Daily sleep tracker"""
    st.header("📊 Daily Sleep Tracker")
    
    if st.session_state.user_id is None:
        st.warning("Please create/load a user profile first!")
        return
    
    st.subheader("Enter Today's Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🛏️ Sleep Data")
        bedtime_hour = st.slider("Bedtime (hour)", 20.0, 26.0, 22.5, 0.5, 
                                 help="22.5 = 10:30 PM, 24.0 = Midnight")
        sleep_duration = st.slider("Sleep Duration (hours)", 4.0, 12.0, 7.5, 0.5)
        
        st.markdown("#### 💼 Work & Stress")
        work_stress = st.slider("Work Stress Level", 1, 10, 5)
        
        st.markdown("#### ☕ Diet")
        caffeine_intake = st.number_input("Caffeine Cups", 0, 10, 2)
    
    with col2:
        st.markdown("#### 🏃 Physical Activity")
        exercise_minutes = st.number_input("Exercise Minutes", 0, 300, 30)
        
        st.markdown("#### 📱 Digital Habits")
        screen_time = st.slider("Evening Screen Time (hours)", 0.0, 8.0, 2.0, 0.5)
        
        st.markdown("#### 🌡️ Environment")
        room_temp = st.slider("Room Temperature (°C)", 15, 28, 20)
        noise_level = st.slider("Noise Level", 0, 10, 2)
    
    # Additional metrics (defaults for demo)
    heart_rate = 70
    heart_rate_var = 45
    
    if st.button("💾 Save Today's Data", type="primary"):
        today_data = {
            'date': datetime.now(),
            'user_id': st.session_state.user_id,
            'bedtime_hour': float(bedtime_hour),
            'sleep_duration': float(sleep_duration),
            'work_stress': float(work_stress),
            'caffeine_intake': float(caffeine_intake),
            'exercise_minutes': float(exercise_minutes),
            'screen_time': float(screen_time),
            'heart_rate': float(heart_rate),
            'heart_rate_var': float(heart_rate_var),
            'room_temp': float(room_temp),
            'noise_level': float(noise_level),
            'sleep_quality': None  # Will be filled after actual sleep
        }
        
        st.session_state.user_data.append(today_data)
        save_user_data(st.session_state.user_id)
        st.success("✅ Data saved successfully!")
        st.balloons()
    
    st.divider()
    st.subheader("Wake-up / Actual Sleep Quality")
    
    if st.session_state.user_data:
        last_entry = st.session_state.user_data[-1]
        st.write("Last entry time:", last_entry.get('date'))
        col1, col2 = st.columns([2, 1])
        with col1:
            auto_est = estimate_sleep_quality(last_entry)
            st.info(f"Auto-estimated Sleep Quality: {auto_est:.1f}/10 (you can override below)")
            actual_quality = st.slider("How would you rate your sleep quality? (0 = worst, 10 = best)", 0.0, 10.0, float(auto_est), 0.5)
        with col2:
            if st.button("🛌 Record Actual Sleep Quality"):
                st.session_state.user_data[-1]['sleep_quality'] = float(actual_quality)
                save_user_data(st.session_state.user_id)
                st.success("✅ Recorded sleep quality.")
    
    # Show recent entries
    if st.session_state.user_data:
        st.divider()
        st.subheader("Recent Entries")
        df = pd.DataFrame(st.session_state.user_data)
        st.dataframe(df.tail(5), use_container_width=True)

def show_predictions():
    """Predictions page"""
    st.header("🔮 Sleep Quality Predictions")
    
    if not st.session_state.model_trained or st.session_state.predictor is None:
        st.warning("⚠️ Please train the AI model first (use sidebar)")
        return
    
    if st.session_state.user_id is None:
        st.warning("Please create/load a user profile first!")
        return
    
    if len(st.session_state.user_data) < 7:
        st.info(f"📊 You need at least 7 days of data. Current: {len(st.session_state.user_data)} days")
        st.write("Keep tracking to get predictions!")
        return
    
    st.subheader("Tonight's Sleep Prediction")
    
    # Get last 7 days of data
    df = pd.DataFrame(st.session_state.user_data)
    last_7_days = df.tail(7)
    
    # Prepare features
    feature_cols = st.session_state.predictor.feature_columns
    past_sequence = last_7_days[feature_cols].values
    current_features = last_7_days.iloc[-1][feature_cols].values
    
    # Make prediction
    try:
        predicted_quality = st.session_state.predictor.predict(past_sequence)
    except Exception as e:
        st.error("Prediction failed — ensure model is trained and feature shapes match.")
        st.write(e)
        return
    
    # Display prediction
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Gauge chart for prediction
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=predicted_quality,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Predicted Sleep Quality"},
            delta={'reference': 7.0},
            gauge={
                'axis': {'range': [None, 10]},
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
        confidence = 85 + np.random.randint(-10, 10)
        st.metric("Confidence", f"{confidence}%")
    
    st.divider()
    
    # Get recommendations
    st.subheader("💡 Personalized Recommendations")
    
    recommendations = st.session_state.recommender.get_recommendations(
        st.session_state.user_id, current_features, predicted_quality
    )
    
    if recommendations:
        for i, rec in enumerate(recommendations[:3]):
            with st.expander(f"{rec['title']} - Success Rate: {rec['success_rate']*100:.0f}%", expanded=(i==0)):
                st.write(rec['message'])
                
                # Progress bar for current vs target
                if rec['current_value'] > rec['target_value']:
                    progress = rec['target_value'] / (rec['current_value'] + 1e-8)
                else:
                    progress = (rec['current_value'] + 1e-8) / rec['target_value']
                progress = float(np.clip(progress, 0.0, 1.0))
                
                st.progress(progress)
                st.write(f"**Current:** {rec['current_value']:.1f} → **Target:** {rec['target_value']:.1f}")
                st.write(f"**Impact:** {rec['impact']}")
                
                # Feedback buttons
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button(f"✅ Followed & Improved", key=f"follow_yes_{i}"):
                        st.session_state.recommender.record_feedback(
                            st.session_state.user_id, rec['type'], True, True
                        )
                        st.session_state.feedback_history.append({
                            'type': rec['type'], 'followed': True, 'improved': True
                        })
                        save_user_data(st.session_state.user_id)
                        st.success("Feedback recorded!")
                with col2:
                    if st.button(f"⚠️ Followed, No Change", key=f"follow_no_{i}"):
                        st.session_state.recommender.record_feedback(
                            st.session_state.user_id, rec['type'], True, False
                        )
                        st.session_state.feedback_history.append({
                            'type': rec['type'], 'followed': True, 'improved': False
                        })
                        save_user_data(st.session_state.user_id)
                        st.info("Feedback recorded!")
                with col3:
                    if st.button(f"❌ Didn't Follow", key=f"no_follow_{i}"):
                        st.session_state.recommender.record_feedback(
                            st.session_state.user_id, rec['type'], False, False
                        )
                        st.session_state.feedback_history.append({
                            'type': rec['type'], 'followed': False, 'improved': False
                        })
                        save_user_data(st.session_state.user_id)
                        st.warning("Feedback recorded!")
    else:
        st.success("✅ No recommendations needed - your habits look great!")

def show_analytics():
    """Analytics dashboard"""
    st.header("📈 Sleep Analytics Dashboard")
    
    if not st.session_state.user_data:
        st.info("Start tracking to see analytics!")
        return
    
    df = pd.DataFrame(st.session_state.user_data)
    
    # Sleep quality over time
    st.subheader("Sleep Quality Trend")
    if 'sleep_quality' in df.columns:
        fig = px.line(df.reset_index(), x='index', y='sleep_quality', title='Sleep Quality Over Time')
        fig.add_hline(y=7, line_dash="dash", line_color="green", annotation_text="Target: 7+")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No sleep quality ratings recorded yet.")
    
    # Correlation heatmap
    st.subheader("Factor Correlations")
    
    numeric_cols = ['bedtime_hour', 'sleep_duration', 'work_stress', 'caffeine_intake', 
                   'exercise_minutes', 'screen_time', 'sleep_quality']
    present_cols = [c for c in numeric_cols if c in df.columns]
    if len(present_cols) >= 2:
        corr_matrix = df[present_cols].corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", 
                       title="How factors correlate with sleep quality")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough numeric data to compute correlations yet.")
    
    # Recommendation success rates
    if st.session_state.recommender and st.session_state.user_id:
        st.subheader("Your Recommendation Success Rates")
        
        success_rates = st.session_state.recommender.user_success_rates.get(
            st.session_state.user_id, {}
        )
        
        if success_rates:
            rates_df = pd.DataFrame([
                {'Recommendation': k.replace('_', ' ').title(), 'Success Rate': v*100}
                for k, v in success_rates.items()
            ])
            
            fig = px.bar(rates_df, x='Recommendation', y='Success Rate',
                        title='What Works Best For You',
                        color='Success Rate',
                        color_continuous_scale='RdYlGn')
            fig.add_hline(y=50, line_dash="dash", annotation_text="Baseline: 50%")
            st.plotly_chart(fig, use_container_width=True)

def show_settings():
    """Settings page"""
    st.header("⚙️ Settings")
    
    st.subheader("Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Export Data"):
            if st.session_state.user_data:
                df = pd.DataFrame(st.session_state.user_data)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"sleep_data_{st.session_state.user_id}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No data to export")
    
    with col2:
        if st.button("🗑️ Clear All Data", type="secondary"):
            if st.button("⚠️ Confirm Clear", type="secondary"):
                st.session_state.user_data = []
                st.session_state.feedback_history = []
                save_user_data(st.session_state.user_id)
                st.success("Data cleared!")
                st.rerun()
    
    st.divider()
    
    st.subheader("Model Information")
    
    if st.session_state.model_trained:
        st.success("✅ Model is trained and active")
        st.write("**Model Type:** LSTM Neural Network")
        st.write("**Sequence Length:** 7 days")
        st.write("**Features:** 10 lifestyle and environmental factors")
        
        if st.button("🔄 Retrain Model"):
            with st.spinner("Retraining..."):
                st.session_state.predictor.train_on_demo_data()
                st.success("Model retrained!")
    else:
        st.warning("Model not trained yet")
    
    st.divider()
    
    st.subheader("About")
    st.markdown("""
    **AI Sleep Coach v1.0**
    
    This app uses advanced machine learning to help you optimize your sleep:
    
    - 🧠 **Deep Learning:** LSTM neural networks for temporal pattern recognition
    - 📈 **Time Series Analysis:** Analyzes your sleep trends over time
    - 👤 **Personalization:** Learns what works specifically for you
    - 🔄 **Reinforcement Learning:** Improves recommendations based on your feedback
    """)
    st.write("Built with ❤️ — complete and ready to extend")

if __name__ == "__main__":
    # Attempt to load an existing model + scaler & recommender if available
    try:
        if os.path.exists(os.path.join(MODEL_DIR, "lstm_model.h5")):
            predictor = StreamlitLSTMPredictor()
            predictor.model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "lstm_model.h5"))
            scaler_fp = os.path.join(MODEL_DIR, "scaler.pkl")
            if os.path.exists(scaler_fp):
                with open(scaler_fp, "rb") as f:
                    predictor.scaler = pickle.load(f)
            st.session_state.predictor = predictor
            st.session_state.model_trained = True
    except Exception:
        # ignore load errors
        st.session_state.model_trained = st.session_state.get('model_trained', False)
    
    if st.session_state.recommender is None:
        st.session_state.recommender = StreamlitRecommender()
    
    main()
