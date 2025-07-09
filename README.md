# Climate Safe Home

A smart, AI-powered web platform that assesses the structural and environmental vulnerability of a home based on image input and geolocation. Designed to help homeowners, builders, and disaster-preparedness authorities make informed safety decisions.

---

## Features

- Upload an image of your home for visual analysis
- Analyzes topography, wind speed, elevation, and climate risks
- GPT-powered insights (fallback to rule-based engine if GPT is unavailable)
- Personalized structural & climate adaptation recommendations
- Cost estimates and DIY guidance
- Generate and download PDF reports
- View builder performance and complaint ratings by location

---

## Tech Stack

| Layer | Tech |
|-------|------|
| Frontend | HTML, CSS, JavaScript |
| Backend | Python, Flask |
| AI | OpenAI GPT-4 (fallback engine with custom rules) |
| Image Processing | OpenCV |
| Climate | OpenWeatherMap API|
| Storage | SQLite |
| Deployment | [Render](https://render.com/) |

---

## Live Demo

🌐 [climatesafehome.onrender.com](https://climatesafehome.onrender.com)

---

## Installation (Local Setup)

```bash
git clone https://github.com/yourusername/climate-safe-home.git
cd climate-safe-home

# Create virtual environment
python -m venv venv
source venv/bin/activate  # (Linux/macOS)
venv\Scripts\activate     # (Windows)

# Install dependencies
pip install -r requirements.txt

# Copy environment variables
cp .env.template .env  # or manually create .env

# Run the app
python start.py
```
--- 

## AI & Fallback Logic
- When OpenAI GPT-4 fails or quota is exceeded:
- A rule-based engine kicks in
- Uses inputs like structure type, house age, roof type, climate risk, and location
- Outputs high-quality, actionable recommendations tailored to the scenario
