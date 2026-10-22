/**
 * Transformer HMS backend
 *  - serves ../frontend
 *  - POST /api/login            role + username + password -> JWT
 *  - reads ESP32 CSV lines from USB serial  (TempC,VoltageV,CurrentA,OilDistanceCM,Fan)
 *  - pushes telemetry to the dashboard over WebSocket (/ws?token=...)
 *  - stores telemetry + events in ./data/*.csv and exports them as .xlsx
 *
 * Run:  SERIAL_PORT=COM5 node server.js        (Windows)
 *       SERIAL_PORT=/dev/ttyUSB0 node server.js (Linux/macOS)
 * Without SERIAL_PORT a built-in simulator streams demo data.
 */
const path = require('path'), fs = require('fs'), crypto = require('crypto'), http = require('http');
const express = require('express'), jwt = require('jsonwebtoken'), XLSX = require('xlsx');
const { WebSocketServer } = require('ws');

const PORT = process.env.PORT || 3000;
const SECRET = process.env.JWT_SECRET || crypto.randomBytes(32).toString('hex');
const SERIAL_PORT = process.env.SERIAL_PORT, BAUD = +process.env.BAUD || 115200;

// Change these (or set PW_AE / PW_AEE / PW_EE environment variables)
const USERS = {
  ae:  { role: 'Assistant Engineer',           pw: process.env.PW_AE  || 'AE@123'  },
  aee: { role: 'Assistant Executive Engineer', pw: process.env.PW_AEE || 'AEE@123' },
  ee:  { role: 'Executive Engineer',           pw: process.env.PW_EE  || 'EE@123'  },
};

const p2 = n => String(n).padStart(2, '0');
const stamp = (d = new Date()) => `${d.getFullYear()}-${p2(d.getMonth() + 1)}-${p2(d.getDate())} ${p2(d.getHours())}:${p2(d.getMinutes())}:${p2(d.getSeconds())}`;
const q = v => '"' + String(v ?? '').replace(/"/g, '""') + '"';
const same = (a, b) => crypto.timingSafeEqual(crypto.createHash('sha256').update(a).digest(), crypto.createHash('sha256').update(b).digest());

// ---------- storage ----------
const DATA = path.join(__dirname, 'data'); fs.mkdirSync(DATA, { recursive: true });
const TEL = path.join(DATA, 'telemetry.csv'), EVT = path.join(DATA, 'events.csv');
if (!fs.existsSync(TEL)) fs.writeFileSync(TEL, 'Timestamp,Temp_C,Voltage_V,Current_A,OilDistance_cm,Fan\n');
if (!fs.existsSync(EVT)) fs.writeFileSync(EVT, 'Timestamp,Severity,Source,Event,User,Temp_C,Voltage_V,Current_A,RUL_years\n');
const tel = [], events = [], CAP = 20000;

// ---------- HTTP ----------
const app = express();
app.use(express.json({ limit: '10kb' }));

app.post('/api/login', (req, res) => {
  const { role, username, password } = req.body || {};
  const u = USERS[role];
  if (!u || !username || typeof password !== 'string' || !same(password, u.pw)) return res.status(401).json({ error: 'Invalid credentials' });
  const token = jwt.sign({ name: String(username).slice(0, 40), role: u.role }, SECRET, { expiresIn: '8h' });
  res.json({ token, role: u.role });
});

const auth = (req, res, next) => {
  try { req.user = jwt.verify((req.headers.authorization || '').replace('Bearer ', '') || req.query.token, SECRET); next(); }
  catch { res.status(401).json({ error: 'Unauthorised' }); }
};

app.get('/api/telemetry', auth, (req, res) => res.json(tel.slice(-Math.min(+req.query.limit || 200, CAP))));
app.get('/api/events', auth, (req, res) => res.json(events.slice(-Math.min(+req.query.limit || 200, CAP))));

app.post('/api/events', auth, (req, res) => {
  const b = req.body || {};
  if (!['INFO', 'WARNING', 'CRITICAL'].includes(b.sev)) return res.status(400).json({ error: 'bad severity' });
  const e = { ts: stamp(), sev: b.sev, src: String(b.src || '').slice(0, 40), msg: String(b.msg || '').slice(0, 300), user: req.user.name,
              T: b.T, V: b.V, I: b.I, rul: b.rul };
  events.push(e); if (events.length > CAP) events.shift();
  fs.appendFile(EVT, [e.ts, e.sev, e.src, e.msg, e.user, e.T, e.V, e.I, e.rul].map(q).join(',') + '\n', () => {});
  res.sendStatus(204);
});

app.get('/api/export', auth, (req, res) => {
  const wb = XLSX.utils.book_new();
  const ev = [['Timestamp', 'Severity', 'Source', 'Event', 'User', 'Temp (°C)', 'Voltage (V)', 'Current (A)', 'RUL (years)']]
    .concat(events.map(e => [e.ts, e.sev, e.src, e.msg, e.user, e.T, e.V, e.I, e.rul]));
  const te = [['Timestamp', 'Temp (°C)', 'Voltage (V)', 'Current (A)', 'Oil distance (cm)', 'Fan']]
    .concat(tel.map(t => [t.ts, t.T, t.V, t.I, t.D, t.fan ? 'ON' : 'OFF']));
  XLSX.utils.book_append_sheet(wb, XLSX.utils.aoa_to_sheet(ev), 'Event Log');
  XLSX.utils.book_append_sheet(wb, XLSX.utils.aoa_to_sheet(te), 'Telemetry');
  res.set({ 'Content-Type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'Content-Disposition': `attachment; filename="Transformer_Log_${stamp().replace(/[: ]/g, '-')}.xlsx"` });
  res.send(XLSX.write(wb, { type: 'buffer', bookType: 'xlsx' }));
});

app.use(express.static(path.join(__dirname, '..', 'frontend')));

// ---------- WebSocket ----------
const server = http.createServer(app);
const wss = new WebSocketServer({ noServer: true });
server.on('upgrade', (req, sock, head) => {
  try {
    const u = new URL(req.url, 'http://localhost');
    if (u.pathname !== '/ws') throw new Error('path');
    jwt.verify(u.searchParams.get('token'), SECRET);
    wss.handleUpgrade(req, sock, head, ws => wss.emit('connection', ws));
  } catch { sock.destroy(); }
});
const broadcast = o => { const m = JSON.stringify(o); wss.clients.forEach(c => c.readyState === 1 && c.send(m)); };

// ---------- telemetry ingest ----------
function ingestLine(line) {
  const p = line.trim().split(',').map(Number);
  if (p.length !== 5 || p.some(Number.isNaN)) return;          // skips the CSV header
  const [T, V, I, D, fan] = p, rec = { type: 'telemetry', ts: stamp(), T, V, I, D, fan };
  tel.push(rec); if (tel.length > CAP) tel.shift();
  fs.appendFile(TEL, [rec.ts, T, V, I, D, fan].join(',') + '\n', () => {});
  broadcast(rec);
}

if (SERIAL_PORT) {
  const { SerialPort, ReadlineParser } = require('serialport');
  const port = new SerialPort({ path: SERIAL_PORT, baudRate: BAUD });
  port.pipe(new ReadlineParser({ delimiter: '\n' })).on('data', ingestLine);
  port.on('open', () => console.log(`Serial ${SERIAL_PORT} open @ ${BAUD}`));
  port.on('error', e => console.error('Serial error:', e.message));
} else {
  console.log('SERIAL_PORT not set - streaming simulated data');
  let t = 0;
  setInterval(() => {
    t += 2;
    const T = 29 + 3 * Math.sin(t / 90) + Math.random() * .4, fan = T > 32 ? 1 : 0;
    ingestLine([T.toFixed(1), (230 + Math.random() * 2).toFixed(2), (2.5 + Math.random() * .2).toFixed(2), (5 + Math.random() * .1).toFixed(2), fan].join(','));
  }, 2000);
}

server.listen(PORT, () => console.log(`Transformer HMS running at http://localhost:${PORT}`));
