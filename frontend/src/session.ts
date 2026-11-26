const SESSION_API_KEY = "nlpulse.session.apiKey";
const SESSION_BASE_URL = "nlpulse.session.baseUrl";

export const DEFAULT_BASE_URL = "http://localhost:8000";

function getStorage(): Storage | null {
  try {
    return typeof window !== "undefined" ? window.sessionStorage : null;
  } catch {
    return null;
  }
}

export function loadSessionApiKey(): string {
  return getStorage()?.getItem(SESSION_API_KEY) ?? "";
}

export function saveSessionApiKey(key: string): void {
  const storage = getStorage();
  if (!storage) return;
  if (key.trim()) {
    storage.setItem(SESSION_API_KEY, key.trim());
  } else {
    storage.removeItem(SESSION_API_KEY);
  }
}

export function clearSessionApiKey(): void {
  getStorage()?.removeItem(SESSION_API_KEY);
}

export function loadSessionBaseUrl(): string {
  return getStorage()?.getItem(SESSION_BASE_URL) ?? DEFAULT_BASE_URL;
}

export function saveSessionBaseUrl(url: string): void {
  const storage = getStorage();
  if (!storage) return;
  const trimmed = url.trim();
  storage.setItem(SESSION_BASE_URL, trimmed || DEFAULT_BASE_URL);
}
