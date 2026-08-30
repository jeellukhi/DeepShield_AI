# DeepShield Final Run Guide

This guide is for running the project on Windows PowerShell.

## 1) Prerequisites

- Python 3.13 installed
- Node.js + npm installed
- MongoDB Community Server running 
- (Optional) Docker Desktop running for container mode

Project folder:

`C:\Users\HP\Desktop\DeepShield_AI`

## 2) Option A: Local Dev Mode (recommended for development)

From PowerShell:

```powershell
cd C:\Users\HP\Desktop\DeepShield_AI
.\scripts\dev-start.ps1
```

If script policy blocks execution:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\dev-start.ps1
```

Expected:

- Backend at `http://localhost:8000`
- Frontend at `http://localhost:5173`

## 3) Option B: Docker Mode

Start (build + run):

```powershell
cd C:\Users\HP\Desktop\DeepShield_AI
.\scripts\docker-start.ps1 -Build
```

Check containers:

```powershell
docker ps
```

Stop containers:

```powershell
.\scripts\docker-stop.ps1
```

You can close Docker Desktop after stopping containers.

## 4) Login and Access

Bootstrap credentials come from `backend/.env`.

- Admin username: `AUTH_ADMIN_USERNAME`
- Admin password: `AUTH_ADMIN_PASSWORD`
- User username: `AUTH_USER_USERNAME`
- User password: `AUTH_USER_PASSWORD`

You can also create a new standard user from the Sign Up page.

## 5) First Admin Workflow

1. Login as admin
2. Open `Admin` page
3. Click `Check Security Status`
4. Click `Check Model Readiness`
5. Click `Check Dataset Quality` and review recommendations
6. If models are missing (or after dataset cleanup):
   - `Train Image Model`
   - `Train Profile Model`
7. Click `Load Thresholds`
8. Keep tuned baseline values:
   - Image fake threshold: `40`
   - Video fake threshold: `55`
   - Profile fake threshold: `40`
9. Click `Save Thresholds`
10. Click `Load Model Evaluation`
11. For full details, see [docs/ACCURACY_UPGRADE_GUIDE.md](/c:/Users/HP/Desktop/DeepShield_AI/docs/ACCURACY_UPGRADE_GUIDE.md)

## 6) User Workflow

1. Login as user (or Sign Up)
2. Use:
   - `Image Check`
   - `Video Check`
   - `Profile Check`
3. Open `Reports` page to view history and details
   - User role is limited to its own report history by account scope.
4. Open `Account` page to change password

## 7) Data and Model Locations

- Dataset images:
  - `ml/datasets/deepfake_images/real`
  - `ml/datasets/deepfake_images/fake`
- Trained models:
  - `ml/models/image_cnn.pkl`
  - `ml/models/profile_nlp.pkl`
- Threshold overrides:
  - `ml/models/thresholds.json`

MongoDB database:

- Database: `deepshield_ai`
- Collections:
  - `analysis_reports`
  - `auth_users`
  - `audit_logs`

## 8) Useful Recovery Commands

Backend direct run:

```powershell
cd C:\Users\HP\Desktop\DeepShield_AI\backend
.\.venv\Scripts\activate
python -m uvicorn app.main:app --reload --port 8000
```

Frontend direct run:

```powershell
cd C:\Users\HP\Desktop\DeepShield_AI\frontend
npm run dev -- --host 0.0.0.0 --port 5173
```

## 9) Common Issues

- `ModuleNotFoundError: No module named 'app'`
  - Run backend command from `backend` folder only.
- `docker ps` cannot connect to daemon
  - Start Docker Desktop and wait for engine ready.
- Button click does nothing in UI
  - Check backend terminal for API error and refresh frontend.
- Export file not appearing
  - Check browser download permission/pop-up block.
