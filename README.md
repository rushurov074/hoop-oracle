# Bracket Brain 🏀
March Madness predictor — Flask backend + polished frontend.

## Run locally
```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```

## Deploy to Render (free)

1. Push this folder to a GitHub repo
   ```bash
   git init
   git add .
   git commit -m "bracket brain"
   git remote add origin https://github.com/YOUR_USERNAME/bracket-brain.git
   git push -u origin main
   ```

2. Go to **render.com** → New → Web Service
3. Connect your GitHub repo
4. Render auto-detects `render.yaml` — just click **Deploy**
5. Your app will be live at `https://bracket-brain.onrender.com` (or similar)

## Project structure
```
madness-app/
├── app.py              # Flask backend + algorithm
├── requirements.txt    # Dependencies
├── render.yaml         # Render deployment config
└── templates/
    └── index.html      # Frontend UI
```

## API endpoints
- `POST /api/predict`      — single game prediction
- `POST /api/simulate`     — full bracket simulation
- `POST /api/montecarlo`   — Monte Carlo simulation
- `GET  /api/teams/sample` — sample bracket teams
