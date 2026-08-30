# DeepShield Final Test Checklist

Mark each item as `PASS`/`FAIL`.

## A) Startup

1. Local start script
   - Action: Run `.\scripts\dev-start.ps1`
   - Expected: Frontend and backend open without crash
2. Docker start script
   - Action: Run `.\scripts\docker-start.ps1 -Build`
   - Expected: 3 containers up (`mongodb`, `backend`, `frontend`)
3. Docker stop script
   - Action: Run `.\scripts\docker-stop.ps1`
   - Expected: No DeepShield containers in `docker ps`

## B) Authentication

1. Admin login
   - Expected: Admin page available
2. User login
   - Expected: Admin page hidden
3. Signup new user
   - Expected: Account created and logged in
4. Change password
   - Expected: Success message and new password works

## C) User Features

1. Image check
   - Expected: Authenticity score, label, fake probability shown
2. Video check
   - Expected: Frames analyzed, warning flags shown (if suspicious)
3. Profile check
   - Expected: Trust score, risk level, recommendation, report ID
4. Reports page
   - Expected: History loads with pagination and detail panel
5. PDF download
   - Expected: Report PDF downloads correctly

## D) Admin Operations

1. Check Security Status
   - Expected: JWT/auth mode, protected routes, auth users visible
2. Check Model Readiness
   - Expected: Image/profile model availability shown
3. Train Image Model
   - Expected: Training result + metrics shown
4. Train Profile Model
   - Expected: Training result + quality gate shown
5. Load Thresholds
   - Expected: Current threshold values loaded
6. Save Thresholds
   - Expected: Overrides saved and returned in response
7. Load Model Evaluation
   - Expected: Accuracy/Precision/Recall/F1 + confusion matrix
8. Load Audit Logs
   - Expected: Recent admin events listed

## E) Data + Persistence

1. MongoDB reports persisted
   - Action: Analyze profile and refresh app
   - Expected: Report still visible in history
2. Threshold persistence
   - Action: Save thresholds and restart backend
   - Expected: Same threshold values remain active
3. Export JSON/CSV
   - Expected: Downloaded files contain report/feedback records

## F) Security and Access

1. User cannot access admin-only endpoints
   - Expected: Forbidden/unauthorized for admin APIs
2. Admin can access admin endpoints
   - Expected: Successful responses
3. Login guard active
   - Expected: Too many bad logins trigger temporary lock

## G) Final Sign-off

Project is sign-off ready when all sections A-F are PASS.
