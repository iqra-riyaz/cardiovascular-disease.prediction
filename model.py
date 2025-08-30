import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import joblib   # <-- for saving the scaler

# Load dataset
df_cardio = pd.read_csv('cardio_train.csv')

# Split semicolon-separated values into columns
df_cardio = df_cardio['id;age;gender;height;weight;ap_hi;ap_lo;cholesterol;gluc;smoke;alco;active;cardio'].str.split(';', expand=True)
df_cardio.columns = ['id', 'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
                     'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio']

# ---- Preprocessing ----
# Convert age from days → years
df_cardio['age'] = (df_cardio['age'].astype(int) / 365.25).round().astype(int)

# Gender: 1 → female, 2 → male → map to {0: female, 1: male}
df_cardio['gender'] = df_cardio['gender'].astype(int).map({1: 0, 2: 1})

# Height & weight for BMI
df_cardio['height'] = df_cardio['height'].astype(float)
df_cardio['weight'] = df_cardio['weight'].astype(float)
df_cardio['bmi'] = df_cardio['weight'] / (df_cardio['height'] / 100)**2

# Cholesterol and glucose: map {1 → 0 (normal), 2/3 → 1 (above normal)}
df_cardio['cholesterol'] = df_cardio['cholesterol'].astype(int).map({1: 0, 2: 1, 3: 1})
df_cardio['gluc'] = df_cardio['gluc'].astype(int).map({1: 0, 2: 1, 3: 1})

# Convert other features to int
df_cardio['ap_hi'] = df_cardio['ap_hi'].astype(int)
df_cardio['ap_lo'] = df_cardio['ap_lo'].astype(int)
df_cardio['smoke'] = df_cardio['smoke'].astype(int)
df_cardio['alco'] = df_cardio['alco'].astype(int)
df_cardio['active'] = df_cardio['active'].astype(int)

# ---- Final Features ----
FEATURES = ['age', 'gender', 'ap_hi', 'ap_lo', 'cholesterol',
            'gluc', 'bmi', 'smoke', 'alco', 'active']

df_processed = df_cardio[FEATURES + ['cardio']].copy()

# Target variable
df_processed['cardio'] = df_processed['cardio'].astype(float)

# ---- Train/Test Split ----
X = df_processed.drop('cardio', axis=1)
y = df_processed['cardio']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- Scaling ----
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ✅ Save scaler for later use in backend
joblib.dump(scaler, "scaler.pkl")
print("✅ Scaler saved as scaler.pkl")

# ---- ANN Model ----
model = Sequential()
model.add(Dense(128, input_shape=(X_train_scaled.shape[1],), activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=128, verbose=0)

# Evaluate
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f'Model Test Accuracy: {accuracy:.4f}')

# Save model
model.save("cardio_model.h5")
print("✅ Model saved as cardio_model.h5")
