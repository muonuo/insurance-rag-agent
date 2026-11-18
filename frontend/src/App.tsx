import { useEffect, useState } from 'react';

import { AdvisorPage } from './AdvisorPage';
import { ApplicationPage } from './ApplicationPage';
import { ApiError, fetchCurrentUser } from './api';
import { clearSession, readSession, saveSession } from './auth';
import { HomePage } from './HomePage';
import type { AuthSession } from './types';

function navigate(path: string) {
  window.history.pushState({}, '', path);
  window.dispatchEvent(new PopStateEvent('popstate'));
}

function applyPath(target: { productId: string; planId: string; itemId: string }) {
  const params = new URLSearchParams({
    product_id: target.productId,
    plan_id: target.planId,
    item_id: target.itemId,
  });
  return `/apply?${params.toString()}`;
}

export default function App() {
  const [path, setPath] = useState(window.location.pathname);
  const [session, setSession] = useState<AuthSession | null>(() => readSession());

  useEffect(() => {
    const updatePath = () => setPath(window.location.pathname);
    window.addEventListener('popstate', updatePath);
    return () => window.removeEventListener('popstate', updatePath);
  }, []);

  // 启动时校验本地缓存的令牌，已过期则立即清理，避免闪现受保护页面
  useEffect(() => {
    const cached = readSession();
    if (!cached) return;
    let ignore = false;
    fetchCurrentUser(cached).catch((reason) => {
      if (!ignore && reason instanceof ApiError && reason.status === 401) {
        clearSession();
        setSession(null);
      }
    });
    return () => {
      ignore = true;
    };
  }, []);

  function authenticated(nextSession: AuthSession) {
    saveSession(nextSession);
    setSession(nextSession);
  }

  function logout() {
    clearSession();
    setSession(null);
    if (path === '/advisor' || path === '/apply') navigate('/');
  }

  function sessionExpired() {
    clearSession();
    setSession(null);
    navigate('/');
  }

  if (path === '/advisor' && session) {
    return (
      <AdvisorPage
        session={session}
        onBack={() => navigate('/')}
        onSessionExpired={sessionExpired}
        onApplyProduct={(target) => navigate(applyPath(target))}
      />
    );
  }

  if (path === '/apply' && session) {
    return (
      <ApplicationPage
        session={session}
        onBack={() => navigate('/advisor')}
        onSessionExpired={sessionExpired}
      />
    );
  }

  return (
    <HomePage
      session={session}
      initialAuthOpen={path === '/advisor' || path === '/apply'}
      onAuthenticated={authenticated}
      onLogout={logout}
      onConsult={() => navigate('/advisor')}
    />
  );
}
