import type {
  AuthApiResponse,
  AuthSession,
  ChatHistory,
  ChatThread,
  InsuranceCategory,
  PlanInterrupt,
  Product,
  RegisterPayload,
  UserSummary,
} from './types';

export const API_BASE = '/api/v1';

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

interface ErrorBody {
  code?: string;
  message?: string;
  detail?: string | Array<{ msg?: string }>;
}

function errorMessage(error: ErrorBody | null): string | undefined {
  if (!error) return undefined;
  // 业务异常：{code, message}
  if (typeof error.message === 'string') return error.message;
  // FastAPI 参数校验：detail 可能是字符串或数组
  if (typeof error.detail === 'string') return error.detail;
  if (Array.isArray(error.detail)) return error.detail[0]?.msg;
  return undefined;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
      ...init?.headers,
    },
  });
  if (!response.ok) {
    const error = await response.json().catch(() => null);
    throw new ApiError(errorMessage(error) ?? '请求失败，请稍后重试', response.status);
  }
  return response.json() as Promise<T>;
}

function toSession(data: AuthApiResponse): AuthSession {
  // 后端snake_case → 前端camelCase
  return {
    token: data.token,
    user: { ...data.user, displayName: data.user.display_name },
  };
}

export async function login(username: string, password: string): Promise<AuthSession> {
  const data = await request<AuthApiResponse>('/auth/login', {
    method: 'POST',
    body: JSON.stringify({ username, password }),
  });
  return toSession(data);
}

export async function register(payload: RegisterPayload): Promise<AuthSession> {
  const data = await request<AuthApiResponse>('/auth/register', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
  return toSession(data);
}

export async function fetchCurrentUser(session: AuthSession): Promise<UserSummary> {
  const user = await request<AuthApiResponse['user']>('/auth/me', {
    headers: { Authorization: `Bearer ${session.token}` },
  });
  return { ...user, displayName: user.display_name };
}

async function authorizedBusinessRequest<T>(
  path: string,
  session: AuthSession,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
      // 保险业务服务的约定头，保持兼容
      'x-user-id': String(session.user.id),
      Authorization: `Bearer ${session.token}`,
      ...init?.headers,
    },
  });
  if (!response.ok) {
    const error = await response.json().catch(() => null);
    throw new ApiError(error?.message ?? '请求失败，请稍后重试', response.status);
  }
  if (response.status === 204) {
    return undefined as T;
  }
  return response.json() as Promise<T>;
}

async function agentRequest<T>(
  path: string,
  session: AuthSession,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${session.token}`,
      ...init?.headers,
    },
  });
  if (!response.ok) {
    const error = await response.json().catch(() => null);
    throw new ApiError(error?.detail ?? error?.message ?? '请求失败，请稍后重试', response.status);
  }
  if (response.status === 204) {
    return undefined as T;
  }
  return response.json() as Promise<T>;
}

export function listProducts(
  category: InsuranceCategory | 'all',
  signal?: AbortSignal,
): Promise<Product[]> {
  const params = new URLSearchParams();
  if (category !== 'all') {
    params.set('category', category);
  }
  const query = params.toString();
  return request<Product[]>(`/products${query ? `?${query}` : ''}`, { signal });
}

export function markPlanItemInsured(
  session: AuthSession,
  planId: string,
  itemId: string,
): Promise<unknown> {
  return authorizedBusinessRequest<unknown>(
    `/insurance-plans/${encodeURIComponent(planId)}/items/${encodeURIComponent(itemId)}/insured`,
    session,
    { method: 'PATCH' },
  );
}

export function getInsurancePlan(session: AuthSession, planId: string): Promise<unknown> {
  return authorizedBusinessRequest<unknown>(
    `/insurance-plans/${encodeURIComponent(planId)}`,
    session,
  );
}

export function listChatThreads(session: AuthSession): Promise<ChatThread[]> {
  return agentRequest<ChatThread[]>('/chat-threads', session);
}

export function createChatThread(session: AuthSession, title = '保险咨询'): Promise<ChatThread> {
  return agentRequest<ChatThread>('/chat-threads', session, {
    method: 'POST',
    body: JSON.stringify({ title }),
  });
}

export function getChatHistory(session: AuthSession, threadId: string): Promise<ChatHistory> {
  return agentRequest<ChatHistory>(`/chat-threads/${threadId}/messages`, session);
}

export function renameChatThread(
  session: AuthSession,
  threadId: string,
  title: string,
): Promise<ChatThread> {
  return agentRequest<ChatThread>(`/chat-threads/${threadId}`, session, {
    method: 'PATCH',
    body: JSON.stringify({ title }),
  });
}

export async function deleteChatThread(session: AuthSession, threadId: string): Promise<void> {
  await agentRequest<void>(`/chat-threads/${threadId}`, session, {
    method: 'DELETE',
  });
}

export async function clearUserProfile(session: AuthSession): Promise<void> {
  await agentRequest<void>('/chat/profile', session, { method: 'DELETE' });
}

export async function sendChatFeedback(
  session: AuthSession,
  threadId: string,
  messageId: string,
  rating: 'up' | 'down',
  comment?: string,
): Promise<void> {
  await agentRequest<void>('/chat/feedback', session, {
    method: 'POST',
    body: JSON.stringify({ thread_id: threadId, message_id: messageId, rating, comment }),
  });
}

export async function streamChat(
  session: AuthSession,
  threadId: string,
  input: { message?: string; decision?: Record<string, unknown> },
  handlers: {
    onMessage: (chunk: string) => void;
    onMeta?: (meta: { message_id?: string }) => void;
    onAdditionalInfo?: (info: unknown) => void;
    onInterrupt?: (interrupt: PlanInterrupt) => void;
  },
): Promise<void> {
  const response = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${session.token}`,
    },
    body: JSON.stringify({ thread_id: threadId, ...input }),
  });
  if (!response.ok || !response.body) {
    const error = await response.json().catch(() => null);
    throw new ApiError(error?.detail ?? error?.message ?? '消息发送失败，请稍后重试', response.status);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split(/\r?\n\r?\n/);
    buffer = parts.pop() ?? '';
    parts.forEach((eventText) => handleSseEvent(eventText, handlers));
  }
  if (buffer.trim()) {
    handleSseEvent(buffer, handlers);
  }
}

function handleSseEvent(
  eventText: string,
  handlers: {
    onMessage: (chunk: string) => void;
    onMeta?: (meta: { message_id?: string }) => void;
    onAdditionalInfo?: (info: unknown) => void;
    onInterrupt?: (interrupt: PlanInterrupt) => void;
  },
) {
  const lines = eventText.split(/\r?\n/);
  const event = lines.find((line) => line.startsWith('event:'))?.slice(6).trim();
  const data = lines
    .filter((line) => line.startsWith('data:'))
    .map((line) => line.slice(5).trimStart())
    .join('\n');
  if (!data) return;
  if (event === 'message') {
    handlers.onMessage(parseSseMessage(data));
    return;
  }
  if (event === 'meta') {
    // 本轮AI消息id，前端反馈时带上它定位消息
    try {
      handlers.onMeta?.(JSON.parse(data));
    } catch {
      // 非法meta直接忽略，不影响消息流
    }
    return;
  }
  if (event === 'interrupt') {
    const interrupt = parseInterrupt(data);
    if (interrupt) handlers.onInterrupt?.(interrupt);
    return;
  }
  if (event === 'additional_info') {
    handlers.onAdditionalInfo?.(parseJsonLike(data) ?? data);
  }
}

function parseSseMessage(data: string): string {
  try {
    const value = JSON.parse(data) as unknown;
    return typeof value === 'string' ? value : JSON.stringify(value);
  } catch {
    return data;
  }
}

function parseInterrupt(data: string): PlanInterrupt | null {
  const parsed = parseJsonLike(data);
  const value = Array.isArray(parsed) ? parsed[0] : parsed;
  if (!value || typeof value !== 'object') return null;
  const candidate = value as Partial<PlanInterrupt>;
  if (!('plan' in candidate) || typeof candidate.confirm_info !== 'string') return null;
  return {
    plan: candidate.plan,
    confirm_info: candidate.confirm_info,
    action: Array.isArray(candidate.action)
      ? candidate.action.filter((item): item is 'approve' | 'reject' =>
          item === 'approve' || item === 'reject',
        )
      : ['approve', 'reject'],
  };
}

function parseJsonLike(data: string): unknown {
  try {
    return JSON.parse(data);
  } catch {
    return null;
  }
}
