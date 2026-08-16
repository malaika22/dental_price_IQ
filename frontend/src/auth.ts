const AUTH_KEY = "dpm_auth_session";
const CRED_USER_KEY = "dpm_cred_user";
const CRED_PASS_KEY = "dpm_cred_pass";

/** Defaults — override with VITE_LOGIN_USER / VITE_LOGIN_PASS */
export const DEFAULT_USER = import.meta.env.VITE_LOGIN_USER || "admin";
export const DEFAULT_PASS = import.meta.env.VITE_LOGIN_PASS || "admin123";

export function getLoginUser(): string {
  try {
    return localStorage.getItem(CRED_USER_KEY) || DEFAULT_USER;
  } catch {
    return DEFAULT_USER;
  }
}

export function getLoginPass(): string {
  try {
    return localStorage.getItem(CRED_PASS_KEY) || DEFAULT_PASS;
  } catch {
    return DEFAULT_PASS;
  }
}

export function hasCustomCredentials(): boolean {
  try {
    return Boolean(localStorage.getItem(CRED_USER_KEY) || localStorage.getItem(CRED_PASS_KEY));
  } catch {
    return false;
  }
}

export function isAuthenticated(): boolean {
  try {
    return localStorage.getItem(AUTH_KEY) === "1";
  } catch {
    return false;
  }
}

export function login(username: string, password: string): boolean {
  if (username.trim() === getLoginUser() && password === getLoginPass()) {
    localStorage.setItem(AUTH_KEY, "1");
    return true;
  }
  return false;
}

export function logout(): void {
  localStorage.removeItem(AUTH_KEY);
}

export function changePassword(currentPassword: string, newPassword: string): string | null {
  if (currentPassword !== getLoginPass()) {
    return "Current password is incorrect.";
  }
  if (newPassword.length < 6) {
    return "New password must be at least 6 characters.";
  }
  if (newPassword === currentPassword) {
    return "New password must be different from the current one.";
  }
  localStorage.setItem(CRED_PASS_KEY, newPassword);
  return null;
}

export function changeUsername(currentPassword: string, newUsername: string): string | null {
  if (currentPassword !== getLoginPass()) {
    return "Current password is incorrect.";
  }
  const trimmed = newUsername.trim();
  if (trimmed.length < 3) {
    return "Username must be at least 3 characters.";
  }
  localStorage.setItem(CRED_USER_KEY, trimmed);
  return null;
}
