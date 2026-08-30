const BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api/v1").replace(/\/$/, "");
const AUTH_STATE_STORAGE_KEY = "deepshield_auth_state";

function getStoredAuthState() {
  if (typeof window === "undefined") return "";
  const raw = window.localStorage.getItem(AUTH_STATE_STORAGE_KEY);
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") return null;
    return parsed;
  } catch {
    return null;
  }
}

function saveAuthState(state) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(AUTH_STATE_STORAGE_KEY, JSON.stringify(state));
}

function getStoredToken() {
  const state = getStoredAuthState();
  return String(state?.access_token || "").trim();
}

function getStoredRefreshToken() {
  const state = getStoredAuthState();
  return String(state?.refresh_token || "").trim();
}

function getAuthHeaders(extra = {}) {
  const token = getStoredToken();
  if (!token) return { ...extra };
  return { ...extra, Authorization: `Bearer ${token}` };
}

async function parseError(res, fallback) {
  let message = fallback;
  try {
    const err = await res.json();
    message = err.detail || message;
  } catch {
    // Keep fallback.
  }
  return message;
}

export function getAuthToken() {
  return getStoredToken();
}

export function setAuthToken(token) {
  const existing = getStoredAuthState() || {};
  const normalized = String(token || "").trim();
  if (!normalized) {
    clearAuthToken();
    return;
  }
  saveAuthState({ ...existing, access_token: normalized });
}

export function clearAuthToken() {
  if (typeof window === "undefined") return;
  window.localStorage.removeItem(AUTH_STATE_STORAGE_KEY);
}

function setAuthTokens(payload) {
  const expiresIn = Number(payload?.expires_in || 0);
  const expiresAt = Date.now() + Math.max(30, expiresIn) * 1000;
  saveAuthState({
    access_token: String(payload?.access_token || ""),
    refresh_token: String(payload?.refresh_token || ""),
    expires_at: expiresAt,
  });
}

async function refreshAuthTokens() {
  const refreshToken = getStoredRefreshToken();
  if (!refreshToken) return false;

  const res = await fetch(`${BASE_URL}/auth/refresh`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ refresh_token: refreshToken }),
  });
  if (!res.ok) {
    clearAuthToken();
    return false;
  }
  const data = await res.json();
  setAuthTokens(data);
  return true;
}

async function authRequest(path, init = {}, retry = true) {
  const withAuth = {
    ...init,
    headers: getAuthHeaders(init.headers || {}),
  };
  let res = await fetch(`${BASE_URL}${path}`, withAuth);
  if (res.status === 401 && retry) {
    const refreshed = await refreshAuthTokens();
    if (refreshed) {
      const retryInit = {
        ...init,
        headers: getAuthHeaders(init.headers || {}),
      };
      res = await fetch(`${BASE_URL}${path}`, retryInit);
    }
  }
  return res;
}

// Backward-compat helpers from previous step (admin key mode).
export function getAdminApiKey() {
  return "";
}

export function setAdminApiKey() {
  // Replaced by JWT login.
}

export async function loginUser(username, password) {
  const res = await fetch(`${BASE_URL}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  if (!res.ok) {
    throw new Error(await parseError(res, "Login failed"));
  }
  const data = await res.json();
  setAuthTokens(data);
  return data;
}

export async function signupUser(username, password) {
  const res = await fetch(`${BASE_URL}/auth/signup`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  if (!res.ok) {
    throw new Error(await parseError(res, "Signup failed"));
  }
  const data = await res.json();
  setAuthTokens(data);
  return data;
}

export async function changePassword(currentPassword, newPassword) {
  const res = await authRequest("/auth/change-password", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      current_password: currentPassword,
      new_password: newPassword,
    }),
  });
  if (!res.ok) {
    throw new Error(await parseError(res, "Password update failed"));
  }
  return res.json();
}

export async function fetchCurrentUser() {
  const res = await authRequest("/auth/me");
  if (!res.ok) {
    throw new Error(await parseError(res, "Session check failed"));
  }
  return res.json();
}

export async function fetchHealth() {
  const res = await fetch(`${BASE_URL}/health`);
  if (!res.ok) throw new Error("Health API failed");
  return res.json();
}

export async function fetchModelReadiness() {
  const res = await authRequest("/model/readiness");
  if (!res.ok) throw new Error(await parseError(res, "Model readiness API failed"));
  return res.json();
}

export async function fetchDatasetQuality(maxScanPerClass = 2000) {
  const safe = Math.max(200, Math.min(10000, Number(maxScanPerClass || 2000)));
  const res = await authRequest(`/dataset/quality?max_scan_per_class=${safe}`);
  if (!res.ok) throw new Error(await parseError(res, "Dataset quality API failed"));
  return res.json();
}

export async function fetchSecurityStatus() {
  const res = await authRequest("/security/status");
  if (!res.ok) throw new Error(await parseError(res, "Security status API failed"));
  return res.json();
}

export async function fetchModelSummary() {
  const res = await authRequest("/model/summary");
  if (!res.ok) throw new Error(await parseError(res, "Model summary API failed"));
  return res.json();
}

export async function fetchModelEvaluation(imageMaxPerClass = 2000, profileMaxSamples = 6000) {
  const params = new URLSearchParams();
  params.set("image_max_per_class", String(imageMaxPerClass));
  params.set("profile_max_samples", String(profileMaxSamples));
  const res = await authRequest(`/model/evaluation?${params.toString()}`);
  if (!res.ok) throw new Error(await parseError(res, "Model evaluation API failed"));
  return res.json();
}

export async function fetchModelThresholds() {
  const res = await authRequest("/model/thresholds");
  if (!res.ok) throw new Error(await parseError(res, "Model thresholds API failed"));
  return res.json();
}

export async function updateModelThresholds(payload) {
  const res = await authRequest("/model/thresholds", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await parseError(res, "Model thresholds update failed"));
  return res.json();
}

export async function fetchProjectStatus() {
  const res = await authRequest("/project/status");
  if (!res.ok) throw new Error(await parseError(res, "Project status API failed"));
  return res.json();
}

export async function fetchDeploymentReadiness() {
  const res = await authRequest("/system/deployment-readiness");
  if (!res.ok) throw new Error(await parseError(res, "Deployment readiness API failed"));
  return res.json();
}

export async function fetchAuditLogs(limit = 50, filters = {}) {
  const params = new URLSearchParams();
  params.set("limit", String(limit));
  if (filters.action?.trim()) {
    params.set("action", filters.action.trim());
  }
  if (filters.actor_username?.trim()) {
    params.set("actor_username", filters.actor_username.trim());
  }
  const res = await authRequest(`/system/audit-logs?${params.toString()}`);
  if (!res.ok) throw new Error(await parseError(res, "Audit logs API failed"));
  return res.json();
}

export async function trainImageModel(maxPerClass = 2000) {
  const res = await authRequest(`/model/train-image?max_per_class=${maxPerClass}`, {
    method: "POST",
  });
  if (!res.ok) {
    throw new Error(await parseError(res, "Model training failed"));
  }
  return res.json();
}

export async function trainProfileModel(maxSamples = 5000) {
  const res = await authRequest(`/model/train-profile?max_samples=${maxSamples}`, {
    method: "POST",
  });
  if (!res.ok) {
    throw new Error(await parseError(res, "Profile model training failed"));
  }
  return res.json();
}

export async function fetchProfileFeedbackStats() {
  const res = await authRequest("/model/profile-feedback-stats");
  if (!res.ok) throw new Error(await parseError(res, "Profile feedback stats API failed"));
  return res.json();
}

export async function exportFeedbackCsv(labeledOnly = true, limit = 10000) {
  const res = await authRequest(
    `/model/export-feedback-csv?labeled_only=${labeledOnly ? "true" : "false"}&limit=${limit}`,
  );
  if (!res.ok) {
    throw new Error(await parseError(res, "Feedback CSV export failed"));
  }

  const blob = await res.blob();
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "deepshield_profile_feedback.csv";
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.URL.revokeObjectURL(url);
}

export async function exportReportsJson(limit = 10000) {
  const res = await authRequest(`/model/export-reports-json?limit=${limit}`);
  if (!res.ok) {
    throw new Error(await parseError(res, "MongoDB JSON export failed"));
  }

  const blob = await res.blob();
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "deepshield_profile_reports.json";
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.URL.revokeObjectURL(url);
}

export async function generateDemoProfiles(count = 25, autoFeedback = true) {
  const res = await authRequest(
    `/demo/generate-profiles?count=${count}&auto_feedback=${autoFeedback ? "true" : "false"}`,
    { method: "POST" },
  );
  if (!res.ok) {
    throw new Error(await parseError(res, "Demo profile generation failed"));
  }
  return res.json();
}

export async function analyzeProfile(payload) {
  const res = await authRequest("/analyze/profile", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await parseError(res, "Analyze API failed"));
  return res.json();
}

export async function analyzeImage(file) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await authRequest("/analyze/image", {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    throw new Error(await parseError(res, "Image analysis failed"));
  }
  return res.json();
}

export async function analyzeVideo(file) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await authRequest("/analyze/video", {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    throw new Error(await parseError(res, "Video analysis failed"));
  }
  return res.json();
}

export async function uploadDatasetImage(file, label) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("label", label);

  const res = await authRequest("/dataset/upload-image", {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    throw new Error(await parseError(res, "Dataset upload failed"));
  }
  return res.json();
}

export async function uploadDatasetZip(file) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await authRequest("/dataset/upload-zip", {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    throw new Error(await parseError(res, "Dataset ZIP import failed"));
  }
  return res.json();
}

export async function fetchReports(limitOrOptions = 10, filtersArg = {}) {
  let page = 1;
  let pageSize = 10;
  let filters = filtersArg || {};

  if (typeof limitOrOptions === "object" && limitOrOptions !== null) {
    page = Number(limitOrOptions.page || 1);
    pageSize = Number(limitOrOptions.page_size || limitOrOptions.pageSize || 10);
    filters = limitOrOptions.filters || {};
  } else {
    pageSize = Number(limitOrOptions || 10);
  }

  const params = new URLSearchParams();
  params.set("limit", String(pageSize));
  params.set("page", String(page));
  if (filters.risk_level && filters.risk_level !== "All") {
    params.set("risk_level", filters.risk_level);
  }
  if (filters.feedback_label && filters.feedback_label !== "All") {
    params.set("feedback_label", filters.feedback_label);
  }
  if (filters.username?.trim()) {
    params.set("username", filters.username.trim());
  }

  const res = await authRequest(`/reports?${params.toString()}`);
  if (!res.ok) throw new Error(await parseError(res, "Reports API failed"));
  return res.json();
}

export async function fetchReportStats(filters = {}) {
  const params = new URLSearchParams();
  if (filters.risk_level && filters.risk_level !== "All") {
    params.set("risk_level", filters.risk_level);
  }
  if (filters.feedback_label && filters.feedback_label !== "All") {
    params.set("feedback_label", filters.feedback_label);
  }
  if (filters.username?.trim()) {
    params.set("username", filters.username.trim());
  }

  const query = params.toString();
  const res = await authRequest(`/reports/stats${query ? `?${query}` : ""}`);
  if (!res.ok) throw new Error(await parseError(res, "Report stats API failed"));
  return res.json();
}

export async function fetchReportDetail(reportId) {
  const res = await authRequest(`/reports/${reportId}`);
  if (!res.ok) {
    throw new Error(await parseError(res, "Report detail API failed"));
  }
  return res.json();
}

export async function updateReportFeedback(reportId, payload) {
  const res = await authRequest(`/reports/${reportId}/feedback`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    throw new Error(await parseError(res, "Report feedback API failed"));
  }
  return res.json();
}

export async function fetchReportTrend(filters = {}, limit = 30) {
  const params = new URLSearchParams();
  params.set("limit", String(limit));
  if (filters.risk_level && filters.risk_level !== "All") {
    params.set("risk_level", filters.risk_level);
  }
  if (filters.feedback_label && filters.feedback_label !== "All") {
    params.set("feedback_label", filters.feedback_label);
  }
  if (filters.username?.trim()) {
    params.set("username", filters.username.trim());
  }

  const res = await authRequest(`/reports/trend?${params.toString()}`);
  if (!res.ok) throw new Error(await parseError(res, "Report trend API failed"));
  return res.json();
}

export async function downloadReportPdf(reportId) {
  const res = await authRequest(`/reports/${reportId}/pdf`);
  if (!res.ok) {
    throw new Error(await parseError(res, "PDF download failed"));
  }

  const blob = await res.blob();
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `deepshield_report_${reportId.slice(0, 8)}.pdf`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.URL.revokeObjectURL(url);
}
