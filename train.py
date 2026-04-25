import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data.csv")

# Combine text
df["text"] = (
    df["question"] + " " +
    df["ideal_answer"] + " " +
    df["student_answer"]
)

X_text = df["text"]
y = df["score"]

# =========================
# TOKENIZATION
# =========================
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_text)

sequences = tokenizer.texts_to_sequences(X_text)

max_len = 30
X = pad_sequences(sequences, maxlen=max_len)

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# MODEL
# =========================
model = Sequential()

model.add(Embedding(5000, 64, input_length=max_len))
model.add(LSTM(64))
model.add(Dropout(0.3))
model.add(Dense(32, activation="relu"))

# Output layer (REGRESSION → score prediction)
model.add(Dense(1, activation="linear"))

# =========================
# COMPILE (FIX FOR KERAS ERROR)
# =========================
model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)

# =========================
# TRAIN
# =========================
model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=8,
    validation_data=(X_test, y_test),
    verbose=1
)

# =========================
# SAVE MODEL (SAFE FORMAT FIX)
# =========================
model.save("model.keras")   # 🔥 FIXED (NOT .h5)

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("Training Completed Successfully 🚀")