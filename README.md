# Task Manager App

A full-stack task management application similar to Jira and Google Tasks, with gamification features and AI-powered assistance.

## 🚀 Features

- **Kanban Board** - Drag and drop task management
- **Gamification** - XP system, levels, and leaderboard
- **AI Assistant** - Smart task suggestions and codebase chat
- **Real-time Collaboration** - Live cursors and presence indicators
- **Dependency Graph** - Visualize task dependencies
- **Analytics Dashboard** - KPI tracking for managers
- **Sprint Planning** - AI-powered sprint management

## 📁 Project Structure

```
├── backend/          # Python FastAPI backend
├── frontend/         # React + Vite frontend
├── docs/             # Documentation files
├── tests/            # Test files
└── run_app.py        # Script to run both frontend and backend
```

## 🛠️ Tech Stack

### Backend
- Python 3.11+
- FastAPI
- Numba (JIT compilation for performance)
- orjson (fast JSON parsing)

### Frontend
- React 18
- Vite
- TailwindCSS

## ⚡ Quick Start

### 1. Install Backend Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Install Frontend Dependencies
```bash
cd frontend
npm install
```

### 3. Run the Application
```bash
python run_app.py
```

Or run separately:
```bash
# Terminal 1 - Backend
cd backend
uvicorn main:app --reload

# Terminal 2 - Frontend
cd frontend
npm run dev
```

## 📖 Documentation

See the `docs/` folder for:
- System Design and Deployment guide
- Code Efficiency Analysis
- Feature Notes
- Git Commands reference

## 🧪 Testing

```bash
cd tests
python test.py
```

## 📝 License

MIT License
