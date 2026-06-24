# Student Career Pathfinder

A full-stack web app that helps students plan their academic journey — recommending next-term courses, predicting their likelihood of success in each, and surfacing AI-generated career-path insights grounded in the courses they've already completed.

Built for **HackUMBC 2025** (Data Science track).

---

## What it does

Students log in with their school ID and land on a personalized dashboard. From there they can:

- **See recommended courses** for the upcoming term, each with a predicted success score based on their completed coursework and prerequisites.
- **Explore career paths** — the app analyzes their course history and returns ranked career options (e.g. Data Scientist, Software Engineer) with a probability, a confidence level, short reasoning, and recommended next courses for that path.
- **Track progress** on a profile dashboard showing total/completed/remaining credits, a completion-progress ring, and a semester-by-semester comparison against the program average.

Advisors get a separate role with access to student analytics and progress tracking.

---

## How it works

**Frontend** — Next.js + TypeScript. Role-based flow (student / advisor), login, dashboard, course recommendations, career-path explorer, and an analytics/profile view with progress charts.

**Backend** — Python. Two prediction paths:

1. **Course recommendation & success prediction.** Course and curriculum data is loaded into a **Neo4j** graph; the backend extracts features from it and uses **neural-net models** to recommend next-term courses and estimate the student's likelihood of success in each.
2. **Career-path exploration.** For open-ended career guidance, the backend calls the **Gemini API** with the student's completed courses and returns ranked career paths with probabilities, confidence levels, reasoning, and suggested follow-on courses.

This split is deliberate: trained models handle the structured recommendation/prediction task, while the LLM handles the more open-ended, reasoning-heavy career-path generation.

---

## Tech stack

| Layer | Technologies |
|---|---|
| Frontend | Next.js, TypeScript, React |
| Backend | Python |
| Data / Graph | Neo4j |
| ML | Neural-net models (course recommendation & success prediction) |
| LLM | Google Gemini API (career-path predictions) |

---

## Getting started

> The app runs locally. You'll need Node.js, Python, a Neo4j instance, and a Gemini API key.

**1. Clone and install frontend dependencies**

```bash
git clone https://github.com/Pranaydeepreddy7017/CareerPathFinder.git
cd CareerPathFinder
npm install
```

**2. Install backend dependencies**

```bash
pip install -r requirements.txt
```

**3. Configure environment**

Create a `.env` (or `.env.local`) with your credentials:

```
GEMINI_API_KEY=your_key_here
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

**4. Load data into Neo4j**

Start your Neo4j instance and load the course dataset (see `core.py` / `service.py` for the ingestion logic).

**5. Run**

```bash
# backend
python api.py

# frontend (in a separate terminal)
npm run dev
```

Then open [http://localhost:3000](http://localhost:3000).

---

## Project layout

```
app/             Next.js frontend (pages, components)
public/          Static assets
api.py           Backend API entry point
service.py       Service / orchestration layer
core.py          Core logic and Neo4j ingestion
career.py        Career-path generation (Gemini)
recommend.py     Course recommendation logic
train.py         Model training for predictions
requirements.txt Python dependencies
```

---

## Notes & limitations

- Built in a hackathon timeframe; some dashboard figures (e.g. the program-average comparison) use mock data for demonstration.
- Not deployed — runs locally.
- The career-path predictions are LLM-generated guidance, not authoritative advising.

---

## Team

Submitted to HackUMBC 2025, Data Science track.
