import { LockOutlined, MailOutlined, UserOutlined } from '@ant-design/icons';
import { Alert, Form, Input, Modal, Tabs } from 'antd';
import { useState } from 'react';

import { login, register } from './api';
import type { AuthSession } from './types';

interface AuthModalProps {
  open: boolean;
  onCancel: () => void;
  onSuccess: (session: AuthSession) => void;
}

type AuthMode = 'login' | 'register';

interface LoginFormValues {
  username: string;
  password: string;
}

interface RegisterFormValues extends LoginFormValues {
  email: string;
  displayName?: string;
}

export function AuthModal({
  open,
  onCancel,
  onSuccess,
}: AuthModalProps) {
  const [form] = Form.useForm<LoginFormValues & RegisterFormValues>();
  const [mode, setMode] = useState<AuthMode>('login');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function switchMode(nextMode: AuthMode) {
    setMode(nextMode);
    setError(null);
    form.resetFields();
  }

  // 关闭后重置状态，避免下次打开时残留上次的错误提示和输入内容
  function afterClose() {
    setMode('login');
    setError(null);
    setLoading(false);
    form.resetFields();
  }

  async function submit(values: LoginFormValues & RegisterFormValues) {
    setLoading(true);
    setError(null);
    try {
      const session =
        mode === 'login'
          ? await login(values.username, values.password)
          : await register({
              username: values.username,
              password: values.password,
              email: values.email,
              display_name: values.displayName?.trim() || undefined,
            });
      onSuccess(session);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '操作失败，请稍后重试');
    } finally {
      setLoading(false);
    }
  }

  const submitLabel = loading
    ? mode === 'login'
      ? '登录中…'
      : '注册中…'
    : mode === 'login'
      ? '登录'
      : '注册';

  return (
    <Modal
      open={open}
      title={null}
      footer={null}
      width={420}
      onCancel={onCancel}
      afterClose={afterClose}
      destroyOnHidden
      centered
    >
      <div className="auth-modal__header">
        <h2>欢迎来到安心保</h2>
        <p>登录后可使用智能顾问服务</p>
      </div>
      <Tabs
        activeKey={mode}
        centered
        items={[
          { key: 'login', label: '登录' },
          { key: 'register', label: '注册' },
        ]}
        onChange={(key) => switchMode(key as AuthMode)}
      />
      {error ? <Alert type="error" showIcon message={error} closable /> : null}
      <Form<LoginFormValues & RegisterFormValues>
        form={form}
        layout="vertical"
        onFinish={submit}
        requiredMark={false}
      >
        <Form.Item
          label="用户名"
          name="username"
          rules={[
            { required: true, message: '请输入用户名' },
            { min: 3, max: 80, message: '用户名长度需在 3-80 位之间' },
          ]}
        >
          <Input prefix={<UserOutlined />} size="large" autoComplete="username" />
        </Form.Item>
        <Form.Item
          label="密码"
          name="password"
          rules={[
            { required: true, message: '请输入密码' },
            { min: 6, max: 64, message: '密码长度需在 6-64 位之间' },
          ]}
        >
          <Input.Password prefix={<LockOutlined />} size="large" autoComplete="current-password" />
        </Form.Item>
        {mode === 'register' ? (
          <>
            <Form.Item
              label="邮箱"
              name="email"
              rules={[
                { required: true, message: '请输入邮箱' },
                { type: 'email', message: '邮箱格式不正确' },
              ]}
            >
              <Input prefix={<MailOutlined />} size="large" autoComplete="email" />
            </Form.Item>
            <Form.Item
              label="昵称（可选）"
              name="displayName"
              rules={[{ max: 100, message: '昵称最多 100 个字符' }]}
            >
              <Input prefix={<UserOutlined />} size="large" />
            </Form.Item>
          </>
        ) : null}
        <button className="auth-submit" type="submit" disabled={loading}>
          {submitLabel}
        </button>
      </Form>
    </Modal>
  );
}
