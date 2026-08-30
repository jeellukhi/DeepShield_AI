import { useEffect, useMemo, useState } from "react";
import {
  analyzeImage,
  analyzeProfile,
  analyzeVideo,
  changePassword,
  clearAuthToken,
  downloadReportPdf,
  fetchAuditLogs,
  fetchDatasetQuality,
  fetchCurrentUser,
  fetchDeploymentReadiness,
  fetchHealth,
  fetchModelEvaluation,
  fetchModelReadiness,
  fetchModelThresholds,
  fetchReportDetail,
  fetchReports,
  fetchSecurityStatus,
  getAuthToken,
  loginUser,
  signupUser,
  trainImageModel,
  trainProfileModel,
  updateModelThresholds,
} from "./services/api";

const initialProfileForm = {
  username: "",
  bio: "",
  followers: 0,
  following: 0,
  posts: 0,
  engagement_rate: 0,
  image_score: 50,
};

const initialPasswordForm = {
  currentPassword: "",
  newPassword: "",
  confirmNewPassword: "",
};

export default function App() {
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [authMode, setAuthMode] = useState("login");
  const [authForm, setAuthForm] = useState({
    username: "",
    password: "",
    confirmPassword: "",
  });
  const [authSession, setAuthSession] = useState(null);
  const [activePage, setActivePage] = useState("home");

  const [health, setHealth] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [imageResult, setImageResult] = useState(null);
  const [imageLoading, setImageLoading] = useState(false);

  const [videoFile, setVideoFile] = useState(null);
  const [videoResult, setVideoResult] = useState(null);
  const [videoLoading, setVideoLoading] = useState(false);

  const [profileForm, setProfileForm] = useState(initialProfileForm);
  const [profileResult, setProfileResult] = useState(null);
  const [profileLoading, setProfileLoading] = useState(false);

  const [reports, setReports] = useState([]);
  const [reportsLoading, setReportsLoading] = useState(false);
  const [reportFilters, setReportFilters] = useState({ username: "", risk_level: "All" });
  const [reportPagination, setReportPagination] = useState({
    page: 1,
    page_size: 12,
    total_count: 0,
    total_pages: 0,
  });
  const [selectedReport, setSelectedReport] = useState(null);
  const [reportDetailLoading, setReportDetailLoading] = useState(false);
  const [passwordForm, setPasswordForm] = useState(initialPasswordForm);
  const [passwordUpdating, setPasswordUpdating] = useState(false);

  const [securityStatus, setSecurityStatus] = useState(null);
  const [modelReadiness, setModelReadiness] = useState(null);
  const [deploymentReadiness, setDeploymentReadiness] = useState(null);
  const [auditLogs, setAuditLogs] = useState([]);
  const [modelEvaluation, setModelEvaluation] = useState(null);
  const [datasetQuality, setDatasetQuality] = useState(null);
  const [modelThresholds, setModelThresholds] = useState(null);
  const [adminLoading, setAdminLoading] = useState({
    security: false,
    readiness: false,
    deployment: false,
    audit: false,
    datasetQuality: false,
    trainImage: false,
    trainProfile: false,
    evaluation: false,
    thresholds: false,
    saveThresholds: false,
  });
  const [trainingConfig, setTrainingConfig] = useState({
    image_max_per_class: 6000,
    profile_max_samples: 12000,
  });
  const [imageTrainingResult, setImageTrainingResult] = useState(null);
  const [profileTrainingResult, setProfileTrainingResult] = useState(null);
  const [thresholdForm, setThresholdForm] = useState({
    image_fake_probability_threshold: 50,
    video_fake_probability_threshold: 55,
    profile_fake_probability_threshold: 50,
  });

  const isAuthenticated = Boolean(authSession);
  const isAdmin = authSession?.role === "admin";

  const navItems = useMemo(() => {
    const base = [
      { id: "home", label: "Home" },
      { id: "image", label: "Image Check" },
      { id: "video", label: "Video Check" },
      { id: "profile", label: "Profile Check" },
      { id: "reports", label: "Reports" },
      { id: "account", label: "Account" },
    ];
    if (isAdmin) {
      base.push({ id: "admin", label: "Admin" });
    }
    return base;
  }, [isAdmin]);
  const activePageLabel = useMemo(
    () => navItems.find((item) => item.id === activePage)?.label || "Home",
    [activePage, navItems]
  );

  useEffect(() => {
    const token = getAuthToken();
    if (!token) return;
    fetchCurrentUser()
      .then((data) => {
        setAuthSession(data.user);
      })
      .catch(() => {
        clearAuthToken();
      });
  }, []);

  useEffect(() => {
    if (!authSession) return;
    if (authSession.role === "user") {
      setProfileForm((prev) => ({ ...prev, username: authSession.username }));
      setReportFilters((prev) => ({ ...prev, username: authSession.username }));
    }
  }, [authSession]);

  function setMessage(nextError = "", nextSuccess = "") {
    setError(nextError);
    setSuccess(nextSuccess);
  }

  function handleAuthInput(event) {
    const { name, value } = event.target;
    setAuthForm((prev) => ({ ...prev, [name]: value }));
  }

  async function handleAuthSubmit(event) {
    event.preventDefault();
    setMessage();

    const username = authForm.username.trim().toLowerCase();
    const password = authForm.password;
    if (!username || !password) {
      setMessage("Username and password are required.");
      return;
    }

    try {
      let data;
      if (authMode === "signup") {
        if (password !== authForm.confirmPassword) {
          setMessage("Password and confirm password must match.");
          return;
        }
        data = await signupUser(username, password);
        setMessage("", "Account created and logged in.");
      } else {
        data = await loginUser(username, password);
        setMessage("", "Login successful.");
      }
      setAuthSession(data.user);
      setAuthForm({ username: username, password: "", confirmPassword: "" });
      setActivePage("home");
    } catch (e) {
      setMessage(e.message);
    }
  }

  function handleLogout() {
    clearAuthToken();
    setAuthSession(null);
    setActivePage("home");
    setProfileResult(null);
    setImageResult(null);
    setVideoResult(null);
    setSelectedReport(null);
    setMessage("", "Logged out.");
  }

  async function checkBackendHealth() {
    setMessage();
    try {
      const data = await fetchHealth();
      setHealth(data);
    } catch (e) {
      setMessage(e.message);
    }
  }

  async function runImageCheck() {
    setMessage();
    if (!imageFile) {
      setMessage("Please choose an image file.");
      return;
    }
    setImageLoading(true);
    try {
      const data = await analyzeImage(imageFile);
      setImageResult(data);
      setProfileForm((prev) => ({ ...prev, image_score: data.authenticity_score }));
      setMessage("", "Image analyzed.");
    } catch (e) {
      setMessage(e.message);
    } finally {
      setImageLoading(false);
    }
  }

  async function runVideoCheck() {
    setMessage();
    if (!videoFile) {
      setMessage("Please choose a video file.");
      return;
    }
    setVideoLoading(true);
    try {
      const data = await analyzeVideo(videoFile);
      setVideoResult(data);
      setProfileForm((prev) => ({ ...prev, image_score: data.authenticity_score }));
      setMessage("", "Video analyzed.");
    } catch (e) {
      setMessage(e.message);
    } finally {
      setVideoLoading(false);
    }
  }

  function handleProfileChange(event) {
    const { name, value } = event.target;
    setProfileForm((prev) => ({ ...prev, [name]: value }));
  }

  async function runProfileCheck() {
    setMessage();
    if (!profileForm.username.trim()) {
      setMessage("Username is required for profile analysis.");
      return;
    }
    setProfileLoading(true);
    try {
      const payload = {
        username: profileForm.username.trim(),
        bio: profileForm.bio.trim(),
        followers: Number(profileForm.followers),
        following: Number(profileForm.following),
        posts: Number(profileForm.posts),
        engagement_rate: Number(profileForm.engagement_rate),
        image_score: Number(profileForm.image_score),
      };
      const data = await analyzeProfile(payload);
      setProfileResult(data);
      setMessage("", "Profile analyzed and report saved.");
    } catch (e) {
      setMessage(e.message);
    } finally {
      setProfileLoading(false);
    }
  }

  function handleReportFilterChange(event) {
    const { name, value } = event.target;
    setReportFilters((prev) => ({ ...prev, [name]: value }));
    setReportPagination((prev) => ({ ...prev, page: 1 }));
  }

  async function loadReportsHistory(targetPage = 1) {
    setMessage();
    setReportsLoading(true);
    try {
      const filters = {
        risk_level: reportFilters.risk_level,
        username: reportFilters.username,
      };
      const data = await fetchReports({
        page: targetPage,
        page_size: reportPagination.page_size,
        filters,
      });
      setReports(data.items || []);
      setReportPagination({
        page: data.page || targetPage,
        page_size: data.page_size || reportPagination.page_size,
        total_count: data.total_count || 0,
        total_pages: data.total_pages || 0,
      });
      setMessage("", `Loaded page ${data.page || targetPage} (${data.count || 0} report(s)).`);
    } catch (e) {
      setMessage(e.message);
    } finally {
      setReportsLoading(false);
    }
  }

  async function viewReport(reportId) {
    setMessage();
    setReportDetailLoading(true);
    try {
      const data = await fetchReportDetail(reportId);
      setSelectedReport(data);
    } catch (e) {
      setMessage(e.message);
    } finally {
      setReportDetailLoading(false);
    }
  }

  function handlePasswordInput(event) {
    const { name, value } = event.target;
    setPasswordForm((prev) => ({ ...prev, [name]: value }));
  }

  async function submitPasswordChange(event) {
    event.preventDefault();
    setMessage();
    if (!passwordForm.currentPassword || !passwordForm.newPassword) {
      setMessage("Current and new password are required.");
      return;
    }
    if (passwordForm.newPassword.length < 8) {
      setMessage("New password must be at least 8 characters.");
      return;
    }
    if (passwordForm.newPassword !== passwordForm.confirmNewPassword) {
      setMessage("New password and confirm password must match.");
      return;
    }
    setPasswordUpdating(true);
    try {
      await changePassword(passwordForm.currentPassword, passwordForm.newPassword);
      setPasswordForm(initialPasswordForm);
      setMessage("", "Password updated successfully.");
    } catch (e) {
      setMessage(e.message);
    } finally {
      setPasswordUpdating(false);
    }
  }

  async function loadSecurity() {
    setAdminLoading((prev) => ({ ...prev, security: true }));
    setMessage();
    try {
      const data = await fetchSecurityStatus();
      setSecurityStatus(data);
    } catch (e) {
      setMessage(e.message);
    } finally {
      setAdminLoading((prev) => ({ ...prev, security: false }));
    }
  }

  async function loadReadiness() {
    setAdminLoading((prev) => ({ ...prev, readiness: true }));
    setMessage();
    try {
      const data = await fetchModelReadiness();
      setModelReadiness(data);
    } catch (e) {
      setMessage(e.message);
    } finally {
      setAdminLoading((prev) => ({ ...prev, readiness: false }));
    }
  }

  async function loadDeployment() {
    setAdminLoading((prev) => ({ ...prev, deployment: true }));
    setMessage();
    try {
      const data = await fetchDeploymentReadiness();
      setDeploymentReadiness(data);
    } catch (e) {
      setMessage(e.message);
    } finally {
      setAdminLoading((prev) => ({ ...prev, deployment: false }));
    }
  }

  async function loadAudit() {
    setAdminLoading((prev) => ({ ...prev, audit: true }));
    setMessage();
    try {
      const data = await fetchAuditLogs(50);
      setAuditLogs(data.items || []);
      setMessage("", `Loaded ${data.count || 0} audit log(s).`);
    } catch (e) {
      setMessage(e.message);
    } finally {
      setAdminLoading((prev) => ({ ...prev, audit: false }));
    }
  }

  async function loadDatasetQuality() {
    setAdminLoading((prev) => ({ ...prev, datasetQuality: true }));
    setMessage();
    try {
      const scanSize = Math.max(200, Number(trainingConfig.image_max_per_class || 2000));
      const data = await fetchDatasetQuality(scanSize);
      setDatasetQuality(data);
      setMessage("", "Dataset quality report loaded.");
    } catch (e) {
      setMessage(e.message);
    } finally {
      setAdminLoading((prev) => ({ ...prev, datasetQuality: false }));
    }
  }

  function handleThresholdInput(event) {
    const { name, value } = event.target;
    setThresholdForm((prev) => ({ ...prev, [name]: value }));
  }

  async function loadThresholds() {
    setAdminLoading((prev) => ({ ...prev, thresholds: true }));
    setMessage();
    try {
      const data = await fetchModelThresholds();
      setModelThresholds(data);
      setThresholdForm({
        image_fake_probability_threshold: data?.values?.image_fake_probability_threshold ?? 50,
        video_fake_probability_threshold: data?.values?.video_fake_probability_threshold ?? 55,
        profile_fake_probability_threshold: data?.values?.profile_fake_probability_threshold ?? 50,
      });
      setMessage("", "Thresholds loaded.");
    } catch (e) {
      setMessage(e.message);
    } finally {
      setAdminLoading((prev) => ({ ...prev, thresholds: false }));
    }
  }

  async function saveThresholds() {
    setAdminLoading((prev) => ({ ...prev, saveThresholds: true }));
    setMessage();
    try {
      const payload = {
        image_fake_probability_threshold: Number(thresholdForm.image_fake_probability_threshold),
        video_fake_probability_threshold: Number(thresholdForm.video_fake_probability_threshold),
        profile_fake_probability_threshold: Number(thresholdForm.profile_fake_probability_threshold),
      };
      const data = await updateModelThresholds(payload);
      setModelThresholds(data);
      setMessage("", "Thresholds updated.");
    } catch (e) {
      setMessage(e.message);
    } finally {
      setAdminLoading((prev) => ({ ...prev, saveThresholds: false }));
    }
  }

  async function loadEvaluation() {
    setAdminLoading((prev) => ({ ...prev, evaluation: true }));
    setMessage();
    try {
      const data = await fetchModelEvaluation(
        Math.max(200, Number(trainingConfig.image_max_per_class || 2000)),
        Math.max(200, Number(trainingConfig.profile_max_samples || 6000)),
      );
      setModelEvaluation(data);
      setMessage("", "Model evaluation loaded.");
    } catch (e) {
      setMessage(e.message);
    } finally {
      setAdminLoading((prev) => ({ ...prev, evaluation: false }));
    }
  }

  function handleTrainingConfigChange(event) {
    const { name, value } = event.target;
    setTrainingConfig((prev) => ({ ...prev, [name]: value }));
  }

  async function runImageTraining() {
    setAdminLoading((prev) => ({ ...prev, trainImage: true }));
    setMessage();
    try {
      const maxPerClass = Math.max(200, Number(trainingConfig.image_max_per_class || 2000));
      const data = await trainImageModel(maxPerClass);
      setImageTrainingResult(data);
      setModelReadiness(data.readiness || null);
      setMessage("", "Image model training completed.");
    } catch (e) {
      setMessage(e.message);
    } finally {
      setAdminLoading((prev) => ({ ...prev, trainImage: false }));
    }
  }

  async function runProfileTraining() {
    setAdminLoading((prev) => ({ ...prev, trainProfile: true }));
    setMessage();
    try {
      const maxSamples = Math.max(500, Number(trainingConfig.profile_max_samples || 5000));
      const data = await trainProfileModel(maxSamples);
      setProfileTrainingResult(data);
      setModelReadiness(data.readiness || null);
      setMessage("", "Profile model training completed.");
    } catch (e) {
      setMessage(e.message);
    } finally {
      setAdminLoading((prev) => ({ ...prev, trainProfile: false }));
    }
  }

  if (!isAuthenticated) {
    return (
      <main className="auth-shell">
        <section className="auth-card">
          <div className="auth-hero">
            <p className="eyebrow">DeepShield AI</p>
            <h1>Verify digital identity before you trust it.</h1>
            <p className="auth-subtitle">
              One workspace for profile screening, image authenticity checks, video frame analysis, and report review.
            </p>
            <div className="auth-highlights">
              <article className="auth-highlight">
                <span className="highlight-index">01</span>
                <div>
                  <h3>Media verification</h3>
                  <p>Upload images or videos and inspect fake-probability signals in seconds.</p>
                </div>
              </article>
              <article className="auth-highlight">
                <span className="highlight-index">02</span>
                <div>
                  <h3>Profile trust scoring</h3>
                  <p>Evaluate suspicious bios, engagement patterns, and account behavior from one form.</p>
                </div>
              </article>
              <article className="auth-highlight">
                <span className="highlight-index">03</span>
                <div>
                  <h3>Role-based access</h3>
                  <p>User and admin flows stay separated so public screens do not expose sensitive controls.</p>
                </div>
              </article>
            </div>
          </div>

          <div className="auth-panel">
            <div className="auth-panel-header">
              <span className="auth-kicker">{authMode === "login" ? "Welcome back" : "Create your workspace"}</span>
              <h2>{authMode === "login" ? "Sign in to continue" : "Set up a new DeepShield account"}</h2>
              <p>
                {authMode === "login"
                  ? "Use your account credentials to access the dashboard."
                  : "Create a standard user account to start running checks and saving reports."}
              </p>
            </div>

            <div className="auth-tabs">
              <button
                className={authMode === "login" ? "tab active" : "tab"}
                onClick={() => setAuthMode("login")}
                type="button"
              >
                Login
              </button>
              <button
                className={authMode === "signup" ? "tab active" : "tab"}
                onClick={() => setAuthMode("signup")}
                type="button"
              >
                Sign Up
              </button>
            </div>

            <form className="auth-form" onSubmit={handleAuthSubmit}>
              <label>
                Username
                <input name="username" value={authForm.username} onChange={handleAuthInput} placeholder="example_user" />
              </label>
              <label>
                Password
                <input type="password" name="password" value={authForm.password} onChange={handleAuthInput} />
              </label>
              {authMode === "signup" && (
                <>
                  <p className="hint">Username: 3-32 chars, letters/numbers/`_`/`.`/`-`. Password: minimum 8 chars.</p>
                  <label>
                    Confirm Password
                    <input
                      type="password"
                      name="confirmPassword"
                      value={authForm.confirmPassword}
                      onChange={handleAuthInput}
                    />
                  </label>
                </>
              )}
              {error && <p className="error">Error: {error}</p>}
              {success && <p className="success">{success}</p>}
              <button type="submit">{authMode === "login" ? "Login" : "Create Account"}</button>
            </form>

            <p className="auth-note">
              Admin credentials are intentionally hidden from the public login screen. Local admin access is configured
              from `backend/.env`.
            </p>
          </div>
        </section>
      </main>
    );
  }

  return (
    <main className="container">
      <header className="topbar">
        <div className="topbar-copy">
          <p className="eyebrow">Threat Review Workspace</p>
          <h1>DeepShield AI</h1>
          <p>Choose one check, analyze it, and review the saved report from a cleaner role-based workflow.</p>
        </div>
        <div className="topbar-side">
          <div className="summary-chips">
            <div className="summary-chip">
              <span className="summary-label">Role</span>
              <strong>{isAdmin ? "Administrator" : "User"}</strong>
            </div>
            <div className="summary-chip">
              <span className="summary-label">Active page</span>
              <strong>{activePageLabel}</strong>
            </div>
          </div>
          <div className="session-chip">
            <span className={`badge ${isAdmin ? "risk-high" : "risk-low"}`}>{isAdmin ? "Admin" : "User"}</span>
            <span>{authSession?.username}</span>
            <button onClick={handleLogout}>Logout</button>
          </div>
        </div>
      </header>

      <section className="card">
        <div className="page-nav">
          {navItems.map((item) => (
            <button
              key={item.id}
              className={activePage === item.id ? "nav-btn active" : "nav-btn"}
              onClick={() => setActivePage(item.id)}
            >
              {item.label}
            </button>
          ))}
          <button className="nav-btn" onClick={checkBackendHealth}>
            Check Health
          </button>
        </div>
      </section>

      {error && <p className="error">Error: {error}</p>}
      {success && <p className="success">{success}</p>}
      {health && (
        <section className="card">
          <h2>Backend Health</h2>
          <pre>{JSON.stringify(health, null, 2)}</pre>
        </section>
      )}

      {activePage === "home" && (
        <section className="card">
          <h2>Choose What You Want To Check</h2>
          <div className="feature-grid">
            <article className="feature-card">
              <h3>Image Authenticity</h3>
              <p>Upload one profile image and detect real/fake signals.</p>
              <button onClick={() => setActivePage("image")}>Open Image Check</button>
            </article>
            <article className="feature-card">
              <h3>Video Authenticity</h3>
              <p>Upload video and run frame-based deepfake risk analysis.</p>
              <button onClick={() => setActivePage("video")}>Open Video Check</button>
            </article>
            <article className="feature-card">
              <h3>Profile Risk Check</h3>
              <p>Analyze username, bio, engagement, and trust score.</p>
              <button onClick={() => setActivePage("profile")}>Open Profile Check</button>
            </article>
            <article className="feature-card">
              <h3>Reports</h3>
              <p>See previous analyses and open full report details.</p>
              <button onClick={() => setActivePage("reports")}>Open Reports</button>
            </article>
          </div>
        </section>
      )}

      {activePage === "image" && (
        <section className="card">
          <h2>Image Check</h2>
          <div className="image-actions">
            <input type="file" accept="image/png,image/jpeg,image/webp" onChange={(e) => setImageFile(e.target.files?.[0] || null)} />
            <button onClick={runImageCheck} disabled={imageLoading}>
              {imageLoading ? "Checking..." : "Analyze Image"}
            </button>
          </div>
          {imageResult && (
            <>
              <div className="result-grid">
                <div className="result-item">
                  <h3>Authenticity Score</h3>
                  <p className="value">{imageResult.authenticity_score}%</p>
                </div>
                <div className="result-item">
                  <h3>Label</h3>
                  <p className={`badge ${imageResult.label === "Real-like" ? "risk-low" : "risk-high"}`}>
                    {imageResult.label}
                  </p>
                </div>
                <div className="result-item">
                  <h3>Fake Probability</h3>
                  <p className="value">{imageResult.fake_probability}%</p>
                </div>
                <div className="result-item">
                  <h3>Model Source</h3>
                  <p>{imageResult.model_source}</p>
                </div>
                {imageResult.raw_fake_probability !== undefined && (
                  <div className="result-item">
                    <h3>Raw Fake Prob.</h3>
                    <p className="value">{imageResult.raw_fake_probability}%</p>
                  </div>
                )}
              </div>
              {imageResult.warning_flags?.length > 0 && (
                <>
                  <h3>Warnings</h3>
                  <ul>
                    {imageResult.warning_flags.map((flag, idx) => (
                      <li key={`${idx}-${flag}`}>{flag}</li>
                    ))}
                  </ul>
                </>
              )}
            </>
          )}
        </section>
      )}

      {activePage === "video" && (
        <section className="card">
          <h2>Video Check</h2>
          <div className="image-actions">
            <input type="file" accept="video/mp4,video/webm,video/quicktime,video/x-msvideo" onChange={(e) => setVideoFile(e.target.files?.[0] || null)} />
            <button onClick={runVideoCheck} disabled={videoLoading}>
              {videoLoading ? "Checking..." : "Analyze Video"}
            </button>
          </div>
          {videoResult && (
            <>
              <div className="result-grid">
                <div className="result-item">
                  <h3>Authenticity Score</h3>
                  <p className="value">{videoResult.authenticity_score}%</p>
                </div>
                <div className="result-item">
                  <h3>Label</h3>
                  <p className={`badge ${videoResult.label === "Real-like" ? "risk-low" : "risk-high"}`}>{videoResult.label}</p>
                </div>
                <div className="result-item">
                  <h3>Fake Probability</h3>
                  <p className="value">{videoResult.fake_probability}%</p>
                </div>
                <div className="result-item">
                  <h3>Frames Analyzed</h3>
                  <p className="value">{videoResult.frames_analyzed}</p>
                </div>
              </div>
              {videoResult.warning_flags?.length > 0 && (
                <>
                  <h3>Warnings</h3>
                  <ul>
                    {videoResult.warning_flags.map((flag, idx) => (
                      <li key={`${idx}-${flag}`}>{flag}</li>
                    ))}
                  </ul>
                </>
              )}
            </>
          )}
        </section>
      )}

      {activePage === "profile" && (
        <section className="card">
          <h2>Profile Check</h2>
          <div className="grid">
            <label>
              Username
              <input name="username" value={profileForm.username} onChange={handleProfileChange} />
            </label>
            <label>
              Followers
              <input type="number" name="followers" min="0" value={profileForm.followers} onChange={handleProfileChange} />
            </label>
            <label>
              Following
              <input type="number" name="following" min="0" value={profileForm.following} onChange={handleProfileChange} />
            </label>
            <label>
              Posts
              <input type="number" name="posts" min="0" value={profileForm.posts} onChange={handleProfileChange} />
            </label>
            <label>
              Engagement Rate (%)
              <input
                type="number"
                name="engagement_rate"
                min="0"
                max="100"
                step="0.1"
                value={profileForm.engagement_rate}
                onChange={handleProfileChange}
              />
            </label>
            <label>
              Image Score (0-100)
              <input
                type="number"
                name="image_score"
                min="0"
                max="100"
                step="0.1"
                value={profileForm.image_score}
                onChange={handleProfileChange}
              />
            </label>
            <label className="bio-row">
              Bio
              <textarea name="bio" value={profileForm.bio} onChange={handleProfileChange} />
            </label>
          </div>
          <button onClick={runProfileCheck} disabled={profileLoading}>
            {profileLoading ? "Analyzing..." : "Analyze Profile"}
          </button>
          {profileResult && (
            <div className="result-grid">
              <div className="result-item">
                <h3>Trust Score</h3>
                <p className="value">{profileResult.trust_score}%</p>
              </div>
              <div className="result-item">
                <h3>Risk Level</h3>
                <p className={`badge risk-${String(profileResult.risk_level || "").toLowerCase()}`}>{profileResult.risk_level}</p>
              </div>
              <div className="result-item">
                <h3>Recommendation</h3>
                <p>{profileResult.recommendation}</p>
              </div>
              <div className="result-item">
                <h3>Report</h3>
                <p>{profileResult.report_id || "N/A"}</p>
              </div>
            </div>
          )}
        </section>
      )}

      {activePage === "reports" && (
        <section className="card">
          <h2>Reports</h2>
          <div className="filters-row">
            {isAdmin ? (
              <label>
                Username
                <input
                  name="username"
                  value={reportFilters.username}
                  onChange={handleReportFilterChange}
                />
              </label>
            ) : (
              <label>
                Username (locked to your account)
                <input name="username" value={reportFilters.username} disabled />
              </label>
            )}
            <label>
              Risk Level
              <select name="risk_level" value={reportFilters.risk_level} onChange={handleReportFilterChange}>
                <option value="All">All</option>
                <option value="Low">Low</option>
                <option value="Medium">Medium</option>
                <option value="High">High</option>
              </select>
            </label>
          </div>
          {!isAdmin && <p className="muted">User role can access only its own report history.</p>}
          <button onClick={() => loadReportsHistory(1)} disabled={reportsLoading}>
            {reportsLoading ? "Loading..." : "Load Reports"}
          </button>
          <p className="muted">
            Total: {reportPagination.total_count} | Page: {reportPagination.page}
            {reportPagination.total_pages ? ` / ${reportPagination.total_pages}` : ""}
          </p>
          {reports.length > 0 && (
            <div className="history-list">
              {reports.map((item) => (
                <article key={item.id} className="history-item">
                  <p>
                    <strong>User:</strong> {item.username}
                  </p>
                  <p>
                    <strong>Trust:</strong> {item.trust_score}%
                  </p>
                  <p>
                    <strong>Risk:</strong> {item.risk_level}
                  </p>
                  <p>
                    <strong>Time:</strong> {item.created_at ? new Date(item.created_at).toLocaleString() : "N/A"}
                  </p>
                  <button onClick={() => viewReport(item.id)} disabled={reportDetailLoading}>
                    {reportDetailLoading ? "Loading..." : "View Detail"}
                  </button>
                  <button onClick={() => downloadReportPdf(item.id)}>Download PDF</button>
                </article>
              ))}
            </div>
          )}
          {reportPagination.total_pages > 1 && (
            <div className="pager-row">
              <button
                onClick={() => loadReportsHistory(Math.max(1, reportPagination.page - 1))}
                disabled={reportsLoading || reportPagination.page <= 1}
              >
                Previous
              </button>
              <button
                onClick={() => loadReportsHistory(reportPagination.page + 1)}
                disabled={reportsLoading || reportPagination.page >= reportPagination.total_pages}
              >
                Next
              </button>
            </div>
          )}
          {selectedReport && (
            <section className="card nested-card">
              <h3>Selected Report Detail</h3>
              <div className="result-grid">
                <div className="result-item">
                  <h3>Trust Score</h3>
                  <p className="value">{selectedReport.trust_score}%</p>
                </div>
                <div className="result-item">
                  <h3>Risk Level</h3>
                  <p className={`badge risk-${String(selectedReport.risk_level || "").toLowerCase()}`}>
                    {selectedReport.risk_level}
                  </p>
                </div>
                <div className="result-item">
                  <h3>Recommendation</h3>
                  <p>{selectedReport.recommendation || "N/A"}</p>
                </div>
                <div className="result-item">
                  <h3>Created</h3>
                  <p>{selectedReport.created_at ? new Date(selectedReport.created_at).toLocaleString() : "N/A"}</p>
                </div>
              </div>
              <div className="detail-grid">
                <article className="detail-card">
                  <h4>Profile Input</h4>
                  <p><strong>Username:</strong> {selectedReport.input_profile?.username || "N/A"}</p>
                  <p><strong>Followers:</strong> {selectedReport.input_profile?.followers ?? "N/A"}</p>
                  <p><strong>Following:</strong> {selectedReport.input_profile?.following ?? "N/A"}</p>
                  <p><strong>Posts:</strong> {selectedReport.input_profile?.posts ?? "N/A"}</p>
                  <p><strong>Engagement:</strong> {selectedReport.input_profile?.engagement_rate ?? "N/A"}%</p>
                  <p><strong>Image Score:</strong> {selectedReport.input_profile?.image_score ?? "N/A"}</p>
                </article>
                <article className="detail-card">
                  <h4>Model Scores</h4>
                  <p><strong>Model Source:</strong> {selectedReport.profile_model_source || "heuristic"}</p>
                  <p><strong>Profile Model Score:</strong> {selectedReport.profile_model_score ?? "N/A"}</p>
                  <p>
                    <strong>Profile Fake Probability:</strong> {selectedReport.profile_model_fake_probability ?? "N/A"}
                  </p>
                  <p><strong>Heuristic Score:</strong> {selectedReport.profile_heuristic_score ?? "N/A"}</p>
                </article>
                <article className="detail-card">
                  <h4>Feedback</h4>
                  <p><strong>Label:</strong> {selectedReport.feedback?.label || "unsure"}</p>
                  <p><strong>Note:</strong> {selectedReport.feedback?.note || "-"}</p>
                  <p>
                    <strong>Updated:</strong>{" "}
                    {selectedReport.feedback?.updated_at
                      ? new Date(selectedReport.feedback.updated_at).toLocaleString()
                      : "N/A"}
                  </p>
                </article>
              </div>
              <article className="detail-card">
                <h4>Risk Factors</h4>
                {selectedReport.risk_factors?.length ? (
                  <ul>
                    {selectedReport.risk_factors.map((factor, idx) => (
                      <li key={`${idx}-${factor}`}>{factor}</li>
                    ))}
                  </ul>
                ) : (
                  <p>No risk factors listed.</p>
                )}
              </article>
            </section>
          )}
        </section>
      )}

      {activePage === "account" && (
        <section className="card">
          <h2>Account Settings</h2>
          <div className="detail-grid">
            <article className="detail-card">
              <h4>Session</h4>
              <p>
                <strong>Username:</strong> {authSession?.username}
              </p>
              <p>
                <strong>Role:</strong> {authSession?.role}
              </p>
            </article>
            <article className="detail-card">
              <h4>Password Policy</h4>
              <p>Use at least 8 characters.</p>
              <p>Do not reuse your old password.</p>
            </article>
          </div>
          <form className="auth-form" onSubmit={submitPasswordChange}>
            <label>
              Current Password
              <input
                type="password"
                name="currentPassword"
                value={passwordForm.currentPassword}
                onChange={handlePasswordInput}
              />
            </label>
            <label>
              New Password
              <input
                type="password"
                name="newPassword"
                value={passwordForm.newPassword}
                onChange={handlePasswordInput}
              />
            </label>
            <label>
              Confirm New Password
              <input
                type="password"
                name="confirmNewPassword"
                value={passwordForm.confirmNewPassword}
                onChange={handlePasswordInput}
              />
            </label>
            <button type="submit" disabled={passwordUpdating}>
              {passwordUpdating ? "Updating..." : "Change Password"}
            </button>
          </form>
        </section>
      )}

      {isAdmin && activePage === "admin" && (
        <section className="card">
          <h2>Admin Console</h2>
          <div className="actions">
            <button onClick={loadSecurity}>{adminLoading.security ? "Checking..." : "Check Security Status"}</button>
            <button onClick={loadReadiness}>{adminLoading.readiness ? "Checking..." : "Check Model Readiness"}</button>
            <button onClick={loadDeployment}>
              {adminLoading.deployment ? "Checking..." : "Check Deployment Readiness"}
            </button>
            <button onClick={loadAudit}>{adminLoading.audit ? "Loading..." : "Load Audit Logs"}</button>
            <button onClick={loadDatasetQuality}>
              {adminLoading.datasetQuality ? "Checking..." : "Check Dataset Quality"}
            </button>
            <button onClick={loadThresholds}>
              {adminLoading.thresholds ? "Loading..." : "Load Thresholds"}
            </button>
            <button onClick={loadEvaluation}>
              {adminLoading.evaluation ? "Evaluating..." : "Load Model Evaluation"}
            </button>
          </div>
          <div className="grid">
            <label>
              Image Max/Class
              <input
                type="number"
                min="200"
                max="20000"
                name="image_max_per_class"
                value={trainingConfig.image_max_per_class}
                onChange={handleTrainingConfigChange}
              />
            </label>
            <label>
              Profile Max Samples
              <input
                type="number"
                min="500"
                max="20000"
                name="profile_max_samples"
                value={trainingConfig.profile_max_samples}
                onChange={handleTrainingConfigChange}
              />
            </label>
          </div>
          <div className="actions">
            <button onClick={runImageTraining} disabled={adminLoading.trainImage}>
              {adminLoading.trainImage ? "Training..." : "Train Image Model"}
            </button>
            <button onClick={runProfileTraining} disabled={adminLoading.trainProfile}>
              {adminLoading.trainProfile ? "Training..." : "Train Profile Model"}
            </button>
          </div>

          <section className="card nested-card">
            <h3>Threshold Tuning</h3>
            <p className="muted">Allowed range: 20 - 80. Lower means more sensitive to fake detection.</p>
            <div className="grid">
              <label>
                Image Fake Threshold (%)
                <input
                  type="number"
                  min="20"
                  max="80"
                  step="0.5"
                  name="image_fake_probability_threshold"
                  value={thresholdForm.image_fake_probability_threshold}
                  onChange={handleThresholdInput}
                />
              </label>
              <label>
                Video Fake Threshold (%)
                <input
                  type="number"
                  min="20"
                  max="80"
                  step="0.5"
                  name="video_fake_probability_threshold"
                  value={thresholdForm.video_fake_probability_threshold}
                  onChange={handleThresholdInput}
                />
              </label>
              <label>
                Profile Fake Threshold (%)
                <input
                  type="number"
                  min="20"
                  max="80"
                  step="0.5"
                  name="profile_fake_probability_threshold"
                  value={thresholdForm.profile_fake_probability_threshold}
                  onChange={handleThresholdInput}
                />
              </label>
            </div>
            <div className="actions">
              <button onClick={saveThresholds} disabled={adminLoading.saveThresholds}>
                {adminLoading.saveThresholds ? "Saving..." : "Save Thresholds"}
              </button>
            </div>
            {modelThresholds && <pre>{JSON.stringify(modelThresholds, null, 2)}</pre>}
          </section>

          {securityStatus && (
            <section className="card nested-card">
              <h3>Security Status</h3>
              <pre>{JSON.stringify(securityStatus, null, 2)}</pre>
            </section>
          )}
          {modelReadiness && (
            <section className="card nested-card">
              <h3>Model Readiness</h3>
              <pre>{JSON.stringify(modelReadiness, null, 2)}</pre>
            </section>
          )}
          {deploymentReadiness && (
            <section className="card nested-card">
              <h3>Deployment Readiness</h3>
              <pre>{JSON.stringify(deploymentReadiness, null, 2)}</pre>
            </section>
          )}
          {datasetQuality && (
            <section className="card nested-card">
              <h3>Dataset Quality</h3>
              {datasetQuality.available ? (
                <>
                  <div className="result-grid">
                    <div className="result-item">
                      <h4>Quality Score</h4>
                      <p className="value">{datasetQuality.quality_score}%</p>
                    </div>
                    <div className="result-item">
                      <h4>Quality Grade</h4>
                      <p>{datasetQuality.quality_grade}</p>
                    </div>
                    <div className="result-item">
                      <h4>Real / Fake</h4>
                      <p>
                        {datasetQuality.counts?.real_total} / {datasetQuality.counts?.fake_total}
                      </p>
                    </div>
                    <div className="result-item">
                      <h4>Imbalance Ratio</h4>
                      <p>{datasetQuality.counts?.imbalance_ratio}</p>
                    </div>
                  </div>
                  {datasetQuality.recommendations?.length > 0 && (
                    <article className="detail-card">
                      <h4>Recommendations</h4>
                      <ul>
                        {datasetQuality.recommendations.map((item, idx) => (
                          <li key={`${idx}-${item}`}>{item}</li>
                        ))}
                      </ul>
                    </article>
                  )}
                </>
              ) : (
                <p>{datasetQuality.message || "Dataset quality report not available."}</p>
              )}
              <pre>{JSON.stringify(datasetQuality, null, 2)}</pre>
            </section>
          )}
          {modelEvaluation && (
            <section className="card nested-card">
              <h3>Model Evaluation</h3>
              <div className="detail-grid">
                <article className="detail-card">
                  <h4>Image Model</h4>
                  {modelEvaluation.image_model?.available ? (
                    <>
                      <p><strong>Type:</strong> {modelEvaluation.image_model.model_type}</p>
                      <p><strong>Threshold:</strong> {modelEvaluation.image_model.threshold_pct}%</p>
                      <p><strong>Accuracy:</strong> {modelEvaluation.image_model.metrics?.accuracy}%</p>
                      <p><strong>Precision:</strong> {modelEvaluation.image_model.metrics?.precision}%</p>
                      <p><strong>Recall:</strong> {modelEvaluation.image_model.metrics?.recall}%</p>
                      <p><strong>F1:</strong> {modelEvaluation.image_model.metrics?.f1_score}%</p>
                      <p>
                        <strong>Confusion:</strong>{" "}
                        TN {modelEvaluation.image_model.confusion_matrix?.tn}, FP {modelEvaluation.image_model.confusion_matrix?.fp}, FN {modelEvaluation.image_model.confusion_matrix?.fn}, TP {modelEvaluation.image_model.confusion_matrix?.tp}
                      </p>
                    </>
                  ) : (
                    <p>{modelEvaluation.image_model?.message || "Image model not available."}</p>
                  )}
                </article>
                <article className="detail-card">
                  <h4>Profile Model</h4>
                  {modelEvaluation.profile_model?.available ? (
                    <>
                      <p><strong>Type:</strong> {modelEvaluation.profile_model.model_type}</p>
                      <p><strong>Threshold:</strong> {modelEvaluation.profile_model.threshold_pct}%</p>
                      <p><strong>Accuracy:</strong> {modelEvaluation.profile_model.metrics?.accuracy}%</p>
                      <p><strong>Precision:</strong> {modelEvaluation.profile_model.metrics?.precision}%</p>
                      <p><strong>Recall:</strong> {modelEvaluation.profile_model.metrics?.recall}%</p>
                      <p><strong>F1:</strong> {modelEvaluation.profile_model.metrics?.f1_score}%</p>
                      <p>
                        <strong>Confusion:</strong>{" "}
                        TN {modelEvaluation.profile_model.confusion_matrix?.tn}, FP {modelEvaluation.profile_model.confusion_matrix?.fp}, FN {modelEvaluation.profile_model.confusion_matrix?.fn}, TP {modelEvaluation.profile_model.confusion_matrix?.tp}
                      </p>
                    </>
                  ) : (
                    <p>{modelEvaluation.profile_model?.message || "Profile model not available."}</p>
                  )}
                </article>
              </div>
            </section>
          )}
          {auditLogs.length > 0 && (
            <section className="card nested-card">
              <h3>Audit Logs</h3>
              <div className="audit-list">
                {auditLogs.map((log) => (
                  <article className="audit-item" key={log.id}>
                    <p>
                      <strong>Action:</strong> {log.action}
                    </p>
                    <p>
                      <strong>Status:</strong> {log.status}
                    </p>
                    <p>
                      <strong>Actor:</strong> {log.actor_username || "unknown"}
                    </p>
                    <p>
                      <strong>Time:</strong> {log.created_at ? new Date(log.created_at).toLocaleString() : "N/A"}
                    </p>
                  </article>
                ))}
              </div>
            </section>
          )}
          {imageTrainingResult && (
            <section className="card nested-card">
              <h3>Image Training Result</h3>
              <pre>{JSON.stringify(imageTrainingResult, null, 2)}</pre>
            </section>
          )}
          {profileTrainingResult && (
            <section className="card nested-card">
              <h3>Profile Training Result</h3>
              <pre>{JSON.stringify(profileTrainingResult, null, 2)}</pre>
            </section>
          )}
        </section>
      )}
    </main>
  );
}
