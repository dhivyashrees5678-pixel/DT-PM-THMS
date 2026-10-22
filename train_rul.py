#!/usr/bin/env python3
"""
Train the hybrid CNN + Bi-LSTM Remaining-Useful-Life model and export it for the dashboard.

  python train_rul.py                      # synthetic ageing dataset (demo)
  python train_rul.py --csv history.csv    # your own labelled history

CSV columns (one row per 2 s sample, in time order):
  T, V, I, oil, fan, rul_years      oil = oil level %, fan = 0/1, rul_years = known remaining life

Output: ../frontend/model/model.json (+ weights) and meta.json. The dashboard loads it automatically.
Use TensorFlow 2.15 / Keras 2 with `pip install tensorflowjs` for a clean conversion.
"""
import argparse, json, os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

W, NOM_V, RATED_I, T_LIMIT, MAX_Y = 16, 230.0, 5.0, 32.0, 25.0   # keep identical to app.js

def feat(T, V, I, oil, fan):
    return np.stack([T / 50, V / NOM_V / 1.3, I / RATED_I / 2, oil / 100, fan], axis=-1)

def stress(T, V, I, oil):
    K = I / RATED_I; HS = T + 22 * K ** 1.6
    th = np.clip((HS - 30) / 25, 0, 1); ov = np.clip((K - 1) / .5, 0, 1)
    vd = np.clip(np.abs(V / NOM_V - 1) / .15, 0, 1); oi = np.clip((85 - oil) / 55, 0, 1)
    return .5 * th + .2 * ov + .15 * vd + .15 * oi

def synthetic(n, rng):
    f = np.linspace(0, 1, W)[None, :]
    col = lambda lo, hi: (lo + (hi - lo) * rng.random(n))[:, None]
    T = col(24, 50) + col(-4, 4) * f + rng.normal(0, .4, (n, W))
    K = col(0, 1.7) * (1 + rng.uniform(-.05, .05, (n, W)))
    V = NOM_V * col(.8, 1.2) * (1 + rng.uniform(-.01, .01, (n, W)))
    drop = (rng.random(n) < .3) * rng.random(n) * 25
    oil = np.clip(col(35, 100) - drop[:, None] * f, 0, 100)
    I = K * RATED_I; fan = (T > T_LIMIT).astype(float)
    s = stress(T, V, I, oil)[:, -4:].mean(axis=1)
    return feat(T, V, I, oil, fan).astype("float32"), np.exp(-3 * s).astype("float32")

def from_csv(path):
    import csv
    rows = list(csv.DictReader(open(path)))
    a = {k: np.array([float(r[k]) for r in rows]) for k in ("T", "V", "I", "oil", "fan", "rul_years")}
    F = feat(a["T"], a["V"], a["I"], a["oil"], a["fan"])
    X = np.stack([F[i:i + W] for i in range(len(F) - W + 1)]).astype("float32")
    y = np.clip(a["rul_years"][W - 1:] / MAX_Y, 0, 1).astype("float32")
    return X, y

def build():
    return models.Sequential([
        layers.Input((W, 5)),
        layers.Conv1D(16, 3, padding="same", activation="relu"),
        layers.Conv1D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling1D(2),
        layers.Bidirectional(layers.LSTM(24)),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv"); ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "frontend", "model"))
    a = ap.parse_args()

    X, y = from_csv(a.csv) if a.csv else synthetic(a.n, np.random.default_rng(1))
    model = build()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    h = model.fit(X, y, epochs=a.epochs, batch_size=64, validation_split=.2, shuffle=True,
                  callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
    mae_years = float(min(h.history["val_mae"]) * MAX_Y)
    print(f"Validation MAE: {mae_years:.2f} years")

    os.makedirs(a.out, exist_ok=True)
    model.save(os.path.join(a.out, "model.keras"))
    import tensorflowjs as tfjs
    tfjs.converters.save_keras_model(model, a.out)
    json.dump({"mae_years": mae_years, "window": W, "samples": int(len(X)), "source": a.csv or "synthetic"},
              open(os.path.join(a.out, "meta.json"), "w"))
    print("Saved to", os.path.abspath(a.out))
