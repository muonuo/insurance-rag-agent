import type { AuthSession } from './types';

const STORAGE_KEY = 'insurance-mall-auth-v3';

export function readSession(): AuthSession | null {
  try {
    const value = window.localStorage.getItem(STORAGE_KEY);
    if (!value) return null;
    const session = JSON.parse(value) as AuthSession;
    // 缺少令牌或用户标识的会话视为无效（兼容历史mock数据）
    if (!session?.token || !session?.user?.id) {
      window.localStorage.removeItem(STORAGE_KEY);
      return null;
    }
    return session;
  } catch {
    window.localStorage.removeItem(STORAGE_KEY);
    return null;
  }
}

export function saveSession(session: AuthSession): void {
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(session));
}

export function clearSession(): void {
  window.localStorage.removeItem(STORAGE_KEY);
}
