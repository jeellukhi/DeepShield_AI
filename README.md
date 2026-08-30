# DeepShield - AI

DeepShield is an AI-powered system for deepfake and fake profile detection.

## Components
- Backend API: FastAPI
- Frontend: React (Vite)
- ML Modules: CNN + NLP pipelines
- Database: MongoDB

## Current Status
- Step 45 completed:
- Profile risk analysis (username/bio/engagement heuristics)
- Image authenticity analysis (OpenCV-based placeholder)
- Video frame authenticity analysis (sampled frames + aggregated score)
- MongoDB report storage and report history API/UI
- PDF report download endpoint and dashboard actions
- Improved recommendation engine with action list
- Report analytics summary and history filters (username/risk)
- Real-time monitoring mode with auto-refresh and live risk alerts
- Explainable risk factors and score contribution breakdown in UI/API
- Report detail endpoint and selectable detailed report view in dashboard
- Trust trend endpoint and timeline chart on dashboard
- Analyst feedback loop (confirmed fake/real/unsure + notes) persisted in MongoDB
- Model readiness endpoint and dashboard card (shows dataset/model availability)
- Dataset upload endpoint + dashboard module for labeled image collection (real/fake)
- Dataset ZIP import endpoint + dashboard flow for existing public datasets (real/fake folders)
- Image model training endpoint + dashboard training action (heuristic -> hybrid mode)
- Video analysis now uses trained image model per frame (with fallback) and shows model source
- Calibrated video scoring using model+heuristic fusion with warning flags to reduce false negatives
- Decision-aware video fake probability now aligned with label threshold, plus raw probability visibility
- Profile NLP model training endpoint added (uses analyst feedback labels) with dashboard action and hybrid scoring
- Profile model quality gate added so weak/imbalanced feedback models are auto-disabled with heuristic fallback
- Legacy profile models without quality gate are now auto-disabled until retrained
- Profile feedback stats endpoint/card added to track remaining confirmed_real/confirmed_fake labels needed for gate
- Quick feedback label buttons added in report cards to speed up analyst review workflow
- Demo profile generator endpoint/button added to create synthetic reports + feedback labels for training prep
- Profile-model explainability fields surfaced in dashboard (model score + fake probability)
- Adaptive profile fusion weights added (model vs heuristic) with explicit transparency in UI/detail
- Model summary endpoint/card added to view trained-at timestamps, metrics, and profile quality-gate metadata
- Feedback CSV export endpoint/button added for labeled profile data backup and offline analysis
- MongoDB-aligned JSON export endpoint/button added for long-term structured report handling
- Added feedback-label filtering (All/Real/Fake/Unsure) across history, analytics stats, and trust trend
- Added a guided Next Action card to recommend the next best dashboard step based on current system state
- Next Action Guide now progresses through operational stages (summary -> history -> trend -> live monitor -> export)
- Added Project Status endpoint/card with MVP readiness, completion percent, checklist, and next-action summary
- Fixed unsure feedback filtering to include both explicit `unsure` and missing feedback records
- Added optional admin-key protection for sensitive endpoints with a security status API and dashboard security controls
- Added deployment-readiness endpoint/card with production checks (APP_ENV, CORS, MongoDB, JWT secret, models)
- Added login page with role-based dashboard access (User/Admin) and improved UI styling
- Added backend JWT login (`/auth/login`, `/auth/me`) with role-based endpoint protection (admin vs user scope)
- Added JWT refresh-token flow (`/auth/refresh`) with frontend auto-refresh/retry on token expiry
- Added MongoDB-backed auth user bootstrap with PBKDF2 password hashing
- Added login brute-force lockout guard (attempt window + lock duration policy)
- Added audit logging for login/refresh/admin-sensitive actions and admin audit-log API/UI
- Added deployment-readiness checks for auth-user availability and lockout policy visibility
- Added startup auto-bootstrap for default auth users to avoid false readiness failures
- Added audit entries for readiness/security/audit-log access actions so admin activity is visible immediately
- Upgraded image model training with texture-stacked features + candidate-model selection + threshold tuning
- Upgraded profile NLP training with engineered profile-signal tokens + candidate-model selection + threshold tuning
- Added user signup API (`/auth/signup`) with secure password hashing and JWT session response
- Rebuilt frontend as page-based flow: Login/Signup -> Home -> Image/Video/Profile/Reports (+ Admin)
- Added in-app password change API/UI (`/auth/change-password`) for authenticated users
- Added paginated reports API/UI (`/reports?page=&limit=`) with page metadata
- Replaced raw report JSON view with structured, readable report detail cards
- Added account settings page for all users (session info + secure password update)
- Added Docker deployment stack for MongoDB + Backend + Frontend
- Added PowerShell helper scripts for local-dev and docker start/stop
- Added admin model-threshold controls (image/video/profile fake-probability thresholds)
- Added admin model-evaluation endpoint/UI with confusion matrix and class-wise metrics
- Upgraded image training pipeline to prevent augmentation leakage (split-first, train-only augmentation)
- Added multicue image features (texture + chroma + frequency/artifact) for better generalization
- Added dataset-quality audit endpoint/UI (balance, blur, corruption, duplicate candidates)
- Added balanced-accuracy and ROC-AUC metrics in model evaluation

## Optional Security Setup
1. Copy `backend/.env.example` to `backend/.env`.
2. Set `AUTH_JWT_SECRET` in `backend/.env` (strong secret).
3. Configure `AUTH_ADMIN_USERNAME` / `AUTH_ADMIN_PASSWORD` and `AUTH_USER_USERNAME` / `AUTH_USER_PASSWORD`.
4. Set token policy (`AUTH_ACCESS_TOKEN_MINUTES`, `AUTH_REFRESH_TOKEN_DAYS`).
5. Set login lockout policy (`AUTH_LOGIN_MAX_ATTEMPTS`, `AUTH_LOGIN_WINDOW_SECONDS`, `AUTH_LOGIN_LOCK_SECONDS`).
6. Restart backend server.
7. Login from dashboard using backend credentials.

Admin endpoints are protected by JWT role checks.

## Deployment Config Basics
- `APP_ENV=development` for local work, `APP_ENV=production` for deployment checks.
- `CORS_ORIGINS` should contain comma-separated trusted frontend URLs.
- Avoid `*` in production CORS.

### Production Template
- Use `backend/.env.production.example` as the base for deployment settings.

### Final Cleanup Applied
- Production CORS locked to deployed frontend domain.
- JWT secret rotated to a strong token format.

## Quick Start Helpers

### Local Dev (Windows PowerShell)
- `.\scripts\dev-start.ps1` -> start backend + frontend in separate terminals.
- `.\scripts\dev-start.ps1 -InstallDeps` -> first-time setup + start.

### Docker
- `.\scripts\docker-start.ps1 -Build` -> build and start full stack.
- `.\scripts\docker-stop.ps1` -> stop full stack.
- Compose file: `docker/docker-compose.yml`
