import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input
from sklearn.preprocessing import LabelEncoder, StandardScaler
import sqlite3
import os
import random
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# Initialize SQLite database
def init_database():
    try:
        conn = sqlite3.connect('user_data.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS user_data (
            user_id INTEGER,
            day INTEGER,
            sleep_hours REAL,
            work_hours REAL,
            meals_per_day INTEGER,
            workout_minutes INTEGER,
            social_media_night INTEGER,
            stress_level INTEGER,
            alcohol_habit INTEGER,
            alcohol_weekly_intake INTEGER,
            smoke_habit INTEGER,
            smoke_weekly_intake INTEGER,
            heart_rate REAL,
            room_temperature REAL,
            sleep_quality INTEGER
        )''')
        c.execute('''CREATE TABLE IF NOT EXISTS user_feedback (
            user_id INTEGER,
            recommendation TEXT,
            feedback INTEGER,
            timestamp TEXT
        )''')
        c.execute('''CREATE TABLE IF NOT EXISTS user_models (
            user_id INTEGER PRIMARY KEY,
            model_path TEXT
        )''')
        conn.commit()
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
        return False
    finally:
        conn.close()
    return True

# Generate dummy data if database is empty
def generate_dummy_data(n_users=10000, days=7):
    try:
        conn = sqlite3.connect('user_data.db')
        data = []
        for user_id in range(n_users):
            for day in range(days):
                alcohol_habit = random.choice([0, 1])
                smoke_habit = random.choice([0, 1])
                data.append({
                    'user_id': user_id,
                    'day': day,
                    'sleep_hours': np.random.uniform(4, 10),
                    'work_hours': np.random.uniform(4, 12),
                    'meals_per_day': np.random.randint(1, 6),
                    'workout_minutes': np.random.randint(0, 121),
                    'social_media_night': random.choice([0, 1]),
                    'stress_level': np.random.randint(0, 11),
                    'alcohol_habit': alcohol_habit,
                    'alcohol_weekly_intake': np.random.randint(1, 21) if alcohol_habit else 0,
                    'smoke_habit': smoke_habit,
                    'smoke_weekly_intake': np.random.randint(1, 141) if smoke_habit else 0,
                    'heart_rate': np.random.uniform(60, 100),
                    'room_temperature': np.random.uniform(18, 28),
                    'sleep_quality': 0
                })
        df = pd.DataFrame(data)
        
        def determine_sleep_quality(row):
            score = 0
            if row['sleep_hours'] >= 7:
                score += 2
            if row['work_hours'] <= 8:
                score += 1
            if row['meals_per_day'] >= 3:
                score += 1
            if row['workout_minutes'] >= 30:
                score += 1
            if row['social_media_night'] == 0:
                score += 1
            if row['stress_level'] <= 5:
                score += 1
            if row['alcohol_weekly_intake'] <= 7:
                score += 1
            if row['smoke_weekly_intake'] <= 20:
                score += 1
            if row['heart_rate'] <= 80:
                score += 1
            if 20 <= row['room_temperature'] <= 24:
                score += 1
            return 1 if score >= 6 else 0
        
        df['sleep_quality'] = df.apply(determine_sleep_quality, axis=1)
        df.to_sql('user_data', conn, if_exists='replace', index=False)
        st.success(f"Generated dummy data for {n_users} users with {days} days each")
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
    finally:
        conn.close()

# Load data for a user or all users
def load_user_data(user_id=None):
    try:
        conn = sqlite3.connect('user_data.db')
        if user_id is not None:
            df = pd.read_sql_query("SELECT * FROM user_data WHERE user_id = ?", conn, params=(user_id,))
        else:
            df = pd.read_sql_query("SELECT * FROM user_data", conn)
        return df
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

# Build LSTM model
def build_lstm_model(input_shape):
    try:
        model = Sequential([
            Input(shape=input_shape),
            LSTM(64, return_sequences=True),
            LSTM(32),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        st.error(f"Error building LSTM model: {e}")
        return None

# Train or load user-specific model
def get_user_model(user_id, X_train, y_train):
    try:
        conn = sqlite3.connect('user_data.db')
        c = conn.cursor()
        c.execute("SELECT model_path FROM user_models WHERE user_id = ?", (user_id,))
        result = c.fetchone()
        
        model = build_lstm_model((7, X_train.shape[2]))
        model_path = f'models/user_{user_id}_model.weights.h5'
        if result and os.path.exists(result[0]):
            model.load_weights(result[0])
            st.info(f"Loaded model for user {user_id}")
        else:
            model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
            os.makedirs('models', exist_ok=True)
            model.save_weights(model_path)
            c.execute("INSERT OR REPLACE INTO user_models (user_id, model_path) VALUES (?, ?)", (user_id, model_path))
            conn.commit()
            st.info(f"Trained and saved new model for user {user_id}")
        return model
    except Exception as e:
        st.error(f"Error in get_user_model: {e}")
        return None
    finally:
        conn.close()

# Contextual bandit for recommendation selection
class ContextualBandit:
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.weights = np.ones((n_actions,))
        self.rewards = [[] for _ in range(n_actions)]
    
    def select_action(self):
        probabilities = self.weights / np.sum(self.weights)
        return np.random.choice(self.n_actions, p=probabilities)
    
    def update(self, action, reward):
        self.rewards[action].append(reward)
        self.weights[action] += reward

# Generate recommendations
def generate_recommendation(user_data, model, bandit, user_id):
    try:
        user_df = load_user_data(user_id)
        if len(user_df) < 7:
            user_df = user_df.reindex(range(7), fill_value=0)
        features = ['sleep_hours', 'work_hours', 'meals_per_day', 'workout_minutes', 'social_media_night', 
                    'stress_level', 'alcohol_habit', 'alcohol_weekly_intake', 'smoke_habit', 
                    'smoke_weekly_intake', 'heart_rate', 'room_temperature']
        X = user_df[features].values[-7:]
        X = np.expand_dims(X, axis=0)
        
        scaler = StandardScaler()
        X = scaler.fit_transform(X.reshape(-1, len(features))).reshape(1, 7, len(features))
        
        prediction = model.predict(X, verbose=0)[0][0]
        sleep_quality = 'good' if prediction > 0.5 else 'poor'
        
        recommendations = [
            "Try to get at least 7 hours of sleep per night.",
            "Reduce work hours or take short breaks to lower stress.",
            "Eat at least 3 balanced meals to stabilize energy levels.",
            "Incorporate at least 30 minutes of exercise daily to improve sleep.",
            "Avoid using Instagram or other social media at night to improve sleep quality.",
            "Practice relaxation techniques (e.g., meditation or deep breathing) to reduce stress.",
            "Reduce alcohol consumption to 7 or fewer drinks per week to improve sleep.",
            "Reduce smoking to 20 or fewer cigarettes per week to improve sleep.",
            "Maintain a resting heart rate below 80 bpm through relaxation or exercise.",
            "Keep room temperature between 20-24°C for optimal sleep."
        ]
        
        action = bandit.select_action()
        selected_recommendation = recommendations[action]
        
        applicable_recommendations = []
        if user_data[0] < 7:
            applicable_recommendations.append(0)
        if user_data[1] > 9:
            applicable_recommendations.append(1)
        if user_data[2] < 3:
            applicable_recommendations.append(2)
        if user_data[3] < 30:
            applicable_recommendations.append(3)
        if user_data[4] == 1:
            applicable_recommendations.append(4)
        if user_data[5] > 5:
            applicable_recommendations.append(5)
        if user_data[6] == 1 and user_data[7] > 7:
            applicable_recommendations.append(6)
        if user_data[8] == 1 and user_data[9] > 20:
            applicable_recommendations.append(7)
        if user_data[10] > 80:
            applicable_recommendations.append(8)
        if user_data[11] < 20 or user_data[11] > 24:
            applicable_recommendations.append(9)
        
        if action not in applicable_recommendations:
            action = random.choice(applicable_recommendations) if applicable_recommendations else 0
            selected_recommendation = recommendations[action]
        
        return [selected_recommendation], sleep_quality, action
    except Exception as e:
        st.error(f"Error generating recommendation: {e}")
        return ["No recommendation due to error"], "unknown", 0

# Save user data to database
def save_user_data(user_id, user_data, sleep_quality):
    try:
        conn = sqlite3.connect('user_data.db')
        c = conn.cursor()
        c.execute("SELECT MAX(day) FROM user_data WHERE user_id = ?", (user_id,))
        last_day = c.fetchone()[0]
        day = (last_day + 1) if last_day is not None else 0
        c.execute("""INSERT INTO user_data (user_id, day, sleep_hours, work_hours, meals_per_day, 
                    workout_minutes, social_media_night, stress_level, alcohol_habit, alcohol_weekly_intake, 
                    smoke_habit, smoke_weekly_intake, heart_rate, room_temperature, sleep_quality) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                  (user_id, day, *user_data, 1 if sleep_quality == 'good' else 0))
        conn.commit()
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
    finally:
        conn.close()

# Save user feedback
def save_user_feedback(user_id, recommendation, feedback):
    try:
        conn = sqlite3.connect('user_data.db')
        c = conn.cursor()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        c.execute("INSERT INTO user_feedback (user_id, recommendation, feedback, timestamp) VALUES (?, ?, ?, ?)",
                  (user_id, recommendation, feedback, timestamp))
        conn.commit()
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
    finally:
        conn.close()

# Streamlit app
def main():
    st.title("Sleep Recommendation System")
    
    if not init_database():
        return
    
    # Check and generate dummy data if needed
    conn = sqlite3.connect('user_data.db')
    df = pd.read_sql_query("SELECT * FROM user_data", conn)
    conn.close()
    if df.empty:
        with st.spinner("Generating dummy data..."):
            generate_dummy_data()
    
    # Load global data for initial training
    df = load_user_data()
    features = ['sleep_hours', 'work_hours', 'meals_per_day', 'workout_minutes', 'social_media_night', 
                'stress_level', 'alcohol_habit', 'alcohol_weekly_intake', 'smoke_habit', 
                'smoke_weekly_intake', 'heart_rate', 'room_temperature']
    X = []
    y = []
    for user_id in df['user_id'].unique():
        user_df = df[df['user_id'] == user_id][features + ['sleep_quality']]
        if len(user_df) >= 7:
            X.append(user_df[features].values[-7:])
            y.append(user_df['sleep_quality'].values[-1])
    
    # Train global model
    try:
        X = np.array(X)
        y = np.array(y)
        scaler = StandardScaler()
        X = scaler.fit_transform(X.reshape(-1, len(features))).reshape(-1, 7, len(features))
        global_model = build_lstm_model((7, len(features)))
        if len(X) > 0:
            global_model.fit(X, y, epochs=10, batch_size=32, verbose=0)
            st.info("Trained global model")
        else:
            st.warning("No sufficient data for global model training")
            global_model = None
    except Exception as e:
        st.error(f"Error training global model: {e}")
        global_model = None
    
    # Initialize contextual bandit
    if 'bandit' not in st.session_state:
        st.session_state.bandit = ContextualBandit(n_actions=10)
    
    # User input form
    st.header("Enter Your Details")
    with st.form("user_input_form"):
        user_id = st.number_input("User ID (0-9999)", min_value=0, max_value=9999, step=1)
        sleep_hours = st.slider("Hours of sleep last night", 0.0, 12.0, 6.0)
        work_hours = st.slider("Hours worked today", 0.0, 16.0, 8.0)
        meals_per_day = st.number_input("Number of meals eaten today", min_value=0, max_value=10, step=1, value=3)
        workout_minutes = st.number_input("Minutes of workout today", min_value=0, max_value=300, step=1, value=0)
        social_media_night = st.selectbox("Used social media at night?", ["No", "Yes"])
        social_media_night = 1 if social_media_night == "Yes" else 0
        stress_level = st.slider("Stress level (0-10)", 0, 10, 5)
        alcohol_habit = st.selectbox("Do you consume alcohol?", ["No", "Yes"])
        alcohol_habit = 1 if alcohol_habit == "Yes" else 0
        alcohol_weekly_intake = st.number_input("Number of alcoholic drinks per week (0-20)", min_value=0, max_value=20, step=1, value=0) if alcohol_habit else 0
        smoke_habit = st.selectbox("Do you smoke?", ["No", "Yes"])
        smoke_habit = 1 if smoke_habit == "Yes" else 0
        smoke_weekly_intake = st.number_input("Number of cigarettes per week (0-140)", min_value=0, max_value=140, step=1, value=0) if smoke_habit else 0
        heart_rate = st.slider("Average resting heart rate today (bpm, 60-100)", 60.0, 100.0, 70.0)
        room_temperature = st.slider("Room temperature (°C, 18-28)", 18.0, 28.0, 22.0)
        submit_button = st.form_submit_button("Get Recommendation")
    
    if submit_button:
        # Validate inputs
        try:
            if stress_level < 0 or stress_level > 10:
                raise ValueError("Stress level must be between 0 and 10")
            if alcohol_weekly_intake < 0 or alcohol_weekly_intake > 20:
                raise ValueError("Alcohol weekly intake must be between 0 and 20")
            if smoke_weekly_intake < 0 or smoke_weekly_intake > 140:
                raise ValueError("Smoke weekly intake must be between 0 and 140")
            if heart_rate < 60 or heart_rate > 100:
                raise ValueError("Heart rate must be between 60 and 100 bpm")
            if room_temperature < 18 or room_temperature > 28:
                raise ValueError("Room temperature must be between 18 and 28°C")
            
            user_data = [sleep_hours, work_hours, meals_per_day, workout_minutes, social_media_night, 
                         stress_level, alcohol_habit, alcohol_weekly_intake, smoke_habit, 
                         smoke_weekly_intake, heart_rate, room_temperature]
            
            # Load or train user-specific model
            user_df = load_user_data(user_id)
            if len(user_df) >= 7:
                X_user = np.array([user_df[features].values[-7:]])
                y_user = np.array([user_df['sleep_quality'].values[-1]])
                X_user = scaler.transform(X_user.reshape(-1, len(features))).reshape(-1, 7, len(features))
                model = get_user_model(user_id, X_user, y_user)
            else:
                model = global_model
                st.warning(f"Using global model for user {user_id} (insufficient data)")
            
            if model is None:
                st.error("No model available. Please try again later.")
                return
            
            # Generate recommendation
            recommendations, sleep_quality, action = generate_recommendation(user_data, model, st.session_state.bandit, user_id)
            
            # Display results
            st.header("Results")
            st.write(f"**Predicted Sleep Quality**: {sleep_quality}")
            st.write("**Recommendation**:")
            for rec in recommendations:
                st.write(f"- {rec}")
            
            # Save user data
            save_user_data(user_id, user_data, sleep_quality)
            
            # Collect feedback
            st.header("Provide Feedback")
            feedback = st.selectbox("Was the recommendation helpful?", ["Select", "Yes", "No"], key="feedback")
            if feedback != "Select":
                feedback_value = 1 if feedback == "Yes" else 0
                save_user_feedback(user_id, recommendations[0], feedback_value)
                st.session_state.bandit.update(action, feedback_value)
                st.success("Feedback submitted!")
                
                # Retrain user model if enough feedback
                conn = sqlite3.connect('user_data.db')
                feedback_df = pd.read_sql_query("SELECT * FROM user_feedback WHERE user_id = ?", conn, params=(user_id,))
                conn.close()
                if len(feedback_df) >= 5:
                    user_df = load_user_data(user_id)
                    if len(user_df) >= 7:
                        X_user = np.array([user_df[features].values[-7:]])
                        y_user = np.array([user_df['sleep_quality'].values[-1]])
                        X_user = scaler.transform(X_user.reshape(-1, len(features))).reshape(-1, 7, len(features))
                        model = get_user_model(user_id, X_user, y_user)
                        st.info("User model retrained based on feedback")
        
        except ValueError as e:
            st.error(f"Invalid input: {e}")

if __name__ == "__main__":
    main()
