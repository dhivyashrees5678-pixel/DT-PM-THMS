# Transformer Digital Twin – Predictive Maintenance

Digital-twin health monitoring for a substation power transformer, with a hybrid CNN + Bi-LSTM
Remaining Useful Life (RUL) model, role-based login and a timestamped event log exported to Excel.

```
transformer-hms/
├── firmware/            your ESP32 sketch (unchanged) – prints TempC,VoltageV,CurrentA,OilDistanceCM,Fan
├── backend/
│   ├── server.js        Express + WebSocket + serial bridge + login + logging + Excel export
│   └── package.json
├── frontend/
│   ├── index.html       login page + dashboard
│   ├── style.css
│   └── app.js           digital twin, CNN + Bi-LSTM (TensorFlow.js), event log, Excel export
└── ml/
    └── train_rul.py     train the same model in Python and export it to the dashboard
```

## 1. Run it (static, no install)
Open `frontend/index.html` in Chrome/Edge. Login is checked in the browser, sensors come from the
simulator, the paste box, or **Connect ESP32** (Web Serial). Model is trained in the browser.

## 2. Run with the backend
```bash
cd backend
npm install
SERIAL_PORT=COM5 node server.js          # Windows: set SERIAL_PORT=COM5 && node server.js
# open http://localhost:3000
```
- Leave `SERIAL_PORT` unset to stream simulated data.
- The ESP32 sketch needs no change: the server reads its CSV line (115200 baud).
- Login is then verified by the server (JWT). Passwords come from `PW_AE`, `PW_AEE`, `PW_EE`
  (defaults `AE@123`, `AEE@123`, `EE@123`). Set `JWT_SECRET` for a stable secret.
- Telemetry and events are saved in `backend/data/*.csv`. `GET /api/export?token=<jwt>` downloads
  a server-side `.xlsx`.

| Endpoint | Purpose |
|---|---|
| `POST /api/login` | `{role: ae\|aee\|ee, username, password}` → `{token}` |
| `GET  /api/telemetry?limit=` | recent samples |
| `GET  /api/events?limit=` | event log |
| `POST /api/events` | add an event (dashboard does this automatically) |
| `GET  /api/export` | Excel with Event Log + Telemetry sheets |
| `WS   /ws?token=` | live telemetry stream |

## 3. Train the model in Python (optional)
```bash
cd ml
pip install "tensorflow==2.15.*" tensorflowjs numpy
python train_rul.py                    # synthetic demo data
python train_rul.py --csv history.csv  # your labelled history (T,V,I,oil,fan,rul_years)
```
This writes `frontend/model/model.json` + weights. The dashboard loads it instead of training in
the browser, and shows the validation MAE.

## Notes
- The synthetic dataset comes from a stress model (thermal, loading, voltage, oil level), so the
  RUL is a demonstrator. For real use, train on labelled history from your own transformers.
- Thresholds are scaled for a bench prototype; set nominal V, rated I and oil distances in the
  calibration row at the top of the dashboard.
- Change the default passwords before any real deployment, and serve over HTTPS.
