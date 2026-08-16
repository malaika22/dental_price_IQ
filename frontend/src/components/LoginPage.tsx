import { useState, type FormEvent } from "react";
import { DEFAULT_PASS, DEFAULT_USER, hasCustomCredentials, login } from "../auth";

interface LoginPageProps {
  onSuccess: () => void;
}

export function LoginPage({ onSuccess }: LoginPageProps) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPass, setShowPass] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [shake, setShake] = useState(false);
  const showDemoHint = !hasCustomCredentials();

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    setError(null);

    if (!username.trim() || !password) {
      setError("Enter your username and password.");
      bumpShake();
      return;
    }

    if (login(username, password)) {
      onSuccess();
      return;
    }

    setError("Invalid username or password.");
    bumpShake();
  };

  const bumpShake = () => {
    setShake(true);
    window.setTimeout(() => setShake(false), 450);
  };

  return (
    <div className="admin-login">
      <div className="admin-login__panel">
        <div className="admin-login__hero">
          <div className="admin-login__mark" aria-hidden>
            <svg viewBox="0 0 40 40" fill="none">
              <rect width="40" height="40" rx="10" fill="#0ea5e9" />
              <path
                d="M12 26c0-5 3.5-9 8-9s8 4 8 9"
                stroke="#fff"
                strokeWidth="2.2"
                strokeLinecap="round"
              />
              <circle cx="20" cy="15" r="3" stroke="#fff" strokeWidth="2.2" />
            </svg>
          </div>
          <p className="admin-login__badge">Admin Portal</p>
          <h1>Dental Price Matcher</h1>
          <p className="admin-login__tagline">
            Manage order analysis, supplier matching, and report history from one place.
          </p>
        </div>

        <div className={`admin-login__card ${shake ? "login-card--shake" : ""}`}>
          <h2>Sign in</h2>
          <p className="admin-login__sub">Use your administrator credentials</p>

          <form className="login-form" onSubmit={handleSubmit} noValidate>
            <label className="login-field">
              <span className="login-field__label">Username</span>
              <input
                className="login-input"
                type="text"
                name="username"
                autoComplete="username"
                placeholder="admin"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                autoFocus
              />
            </label>

            <label className="login-field">
              <span className="login-field__label">Password</span>
              <div className="login-input-wrap">
                <input
                  className="login-input"
                  type={showPass ? "text" : "password"}
                  name="password"
                  autoComplete="current-password"
                  placeholder="••••••••"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                />
                <button
                  type="button"
                  className="login-reveal"
                  onClick={() => setShowPass((v) => !v)}
                  aria-label={showPass ? "Hide password" : "Show password"}
                >
                  {showPass ? "Hide" : "Show"}
                </button>
              </div>
            </label>

            {error && (
              <div className="alert alert--error login-form__error" role="alert">
                {error}
              </div>
            )}

            <button type="submit" className="btn btn--primary btn--lg login-form__submit">
              Enter portal
            </button>
          </form>

          {showDemoHint ? (
            <p className="login-hint">
              Demo: <code>{DEFAULT_USER}</code> / <code>{DEFAULT_PASS}</code>
            </p>
          ) : (
            <p className="login-hint">Use the credentials set in Admin → Settings.</p>
          )}
        </div>
      </div>
    </div>
  );
}
