'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

type Role = 'user' | 'assistant';

type Product = {
  part_type?: string;
  part_number?: string;
  man_part_number?: string;
  price?: number;
  brand?: string;
  stock_status?: string;
  appliance_type?: string;
  symptoms?: string[];
  replaces_part_numbers?: string[];
  installation_video_url?: string;
  item_url?: string;
};

type Message = {
  id: string;
  role: Role;
  content: string;
  products?: Product[];
  queryType?: string;
};


type RunMeta = {
  status: 'idle' | 'streaming' | 'done' | 'error';
  queryType?: string;
  productsCount?: number;
  errorMessage?: string;
};

function uid() {
  return Math.random().toString(16).slice(2) + Date.now().toString(16);
}

function safeJsonParse(s: string) {
  try {
    return JSON.parse(s);
  } catch {
    return null;
  }
}

export default function Page() {
  const API_BASE = useMemo(() => 'http://localhost:8000', []);

  const [messages, setMessages] = useState<Message[]>([
    {
      id: uid(),
      role: 'assistant',
      content:
        "Hi! I’m your PartSelect assistant.\n\nI can help with **refrigerator** and **dishwasher** parts: product lookup, installation guidance, compatibility questions, and troubleshooting.\n\nAsk me anything about these appliances!",
    },
  ]);

  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [online, setOnline] = useState<boolean | null>(null);
  const [runMeta, setRunMeta] = useState<RunMeta>({
    status: 'idle',
    productsCount: 0,
    queryType: undefined,
  });

  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const r = await fetch(`${API_BASE}/health`, { cache: 'no-store' });
        if (!cancelled) setOnline(r.ok);
      } catch {
        if (!cancelled) setOnline(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [API_BASE]);

  const examples = [
  'My refrigerator is leaking water - what are the most common causes and parts to check?',
  'Is part number PS11734579 compatible with my model WAX450SAES9?',
  'My dishwasher isn’t cleaning dishes well - what should I troubleshoot first?',
];


  const sendQuick = (text: string) => setInput(text);

  const resetChat = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/reset`, { method: 'POST' });
      const data = await res.json();
      const intro = data?.intro || 'Hi! How can I help?';
      setMessages([{ id: uid(), role: 'assistant', content: intro }]);
      setRunMeta({ status: 'idle', productsCount: 0, queryType: undefined });
      setInput('');
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          id: uid(),
          role: 'assistant',
          content:
            "I couldn’t reset from the server. No worries — ask your next question and we’ll continue.",
        },
      ]);
    }
  };

  const sendMessage = async () => {
    const trimmed = input.trim();
    if (!trimmed || loading) return;

    const userMsg: Message = { id: uid(), role: 'user', content: trimmed };
    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setLoading(true);
    setRunMeta({ status: 'streaming', queryType: undefined, productsCount: 0 });

    const assistantId = uid();
    setMessages((prev) => [...prev, { id: assistantId, role: 'assistant', content: '' }]);

    const historyPayload = [...messages, userMsg].map((m) => ({
      role: m.role,
      content: m.content,
    }));

    try {
      const res = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: trimmed, history: historyPayload }),
      });

      if (!res.ok || !res.body) throw new Error(`Request failed (${res.status})`);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();

      let buffer = '';
      let fullText = '';
      let productsForThisAnswer: Product[] = [];
      let queryType: string | undefined;

      const handleLine = (line: string) => {
        if (!line.startsWith('data: ')) return;
        const payload = safeJsonParse(line.slice(6));
        if (!payload) return;

        if (payload.type === 'chunk') {
          fullText += payload.content ?? '';
          setMessages((prev) =>
            prev.map((m) => (m.id === assistantId ? { ...m, content: fullText } : m))
          );
          return;
        }

        if (payload.type === 'products') {
          productsForThisAnswer = Array.isArray(payload.products) ? payload.products : [];
          setRunMeta((prev) => ({ ...prev, productsCount: productsForThisAnswer.length }));
          return;
        }

        if (payload.type === 'metadata') {
          queryType = payload.query_type || '';
          setRunMeta((prev) => ({ ...prev, queryType }));

          setMessages((prev) =>
            prev.map((m) => (m.id === assistantId ? { ...m, queryType } : m))
          );
          return;
        }


        if (payload.type === 'error') {
          setRunMeta({ status: 'error', errorMessage: payload.message || 'Unknown error' });
        }
      };

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        let idx;
        while ((idx = buffer.indexOf('\n\n')) !== -1) {
          const rawEvent = buffer.slice(0, idx);
          buffer = buffer.slice(idx + 2);
          const lines = rawEvent.split('\n').map((l) => l.trim());
          for (const line of lines) handleLine(line);
        }
      }

      if (productsForThisAnswer.length > 0) {
        setMessages((prev) =>
          prev.map((m) => (m.id === assistantId ? { ...m, products: productsForThisAnswer } : m))
        );
      }

      setRunMeta((prev) => ({ ...prev, status: 'done', queryType }));
    } catch (e: any) {
      setRunMeta({ status: 'error', errorMessage: e?.message || 'Failed to send message' });
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? {
                ...m,
                content:
                  "Sorry — I hit an error calling the agent.\n\nMake sure the backend is running on **http://localhost:8000** and try again.",
              }
            : m
        )
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="brandRow">
          <div className="brand">PartSelect Agent</div>
          <div className="pill">
            <span className={`dot ${online === null ? '' : online ? 'ok' : 'bad'}`} />
            {online === null ? 'Checking…' : online ? 'Online' : 'Offline'}
          </div>
        </div>

        <div className="card">
          <div className="cardTitle">Last run</div>
          <div className="kv">
            <span>Status</span>
            <span>
              {runMeta.status === 'idle'
                ? 'Idle'
                : runMeta.status === 'streaming'
                ? 'Responding…'
                : runMeta.status === 'done'
                ? 'Done'
                : 'Error'}
            </span>
          </div>
          <div className="kv">
            <span>Query type</span>
            <span>{runMeta.queryType || '—'}</span>
          </div>
          <div className="kv">
            <span>Products</span>
            <span>{runMeta.productsCount ?? 0}</span>
          </div>

          {runMeta.status === 'error' && runMeta.errorMessage ? (
            <div style={{ marginTop: 10, color: 'rgba(255,120,120,.95)', fontSize: 12 }}>
              {runMeta.errorMessage}
            </div>
          ) : null}

          <button className="btn" onClick={resetChat}>
            New chat
          </button>
        </div>

        <div className="examples">
          <h3>Example queries</h3>
          {examples.map((q) => (
            <button key={q} className="exBtn" onClick={() => sendQuick(q)}>
              {q}
            </button>
          ))}
        </div>

        <div style={{ marginTop: 16, fontSize: 12, color: 'rgba(255,255,255,.55)' }}>
          Scope: refrigerator + dishwasher parts only.
        </div>
      </aside>

      {/* Main */}
      <main className="main">
        {/* Mobile topbar */}
        <div className="topbar">
          <div className="topbarTitle">PartSelect Agent</div>
          <button className="topbarBtn" onClick={resetChat}>
            New chat
          </button>
        </div>

        {/* Chat */}
        <div className="chat">
          {messages.map((m) => (
            <div key={m.id} className={`row ${m.role}`}>
              <div className={`bubble ${m.role}`}>
                {m.role === 'assistant' ? (
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={{
                      a: ({ href, children }) => (
                        <a
                          href={href}
                          target="_blank"
                          rel="noreferrer"
                          style={{
                            color: 'rgba(147,197,253,.95)',
                            fontWeight: 900,
                            textDecoration: 'underline',
                          }}
                        >
                          {children}
                        </a>
                      ),
                      p: ({ children }) => <p style={{ margin: '8px 0' }}>{children}</p>,
                      ul: ({ children }) => <ul style={{ paddingLeft: 18, margin: '8px 0' }}>{children}</ul>,
                      li: ({ children }) => <li style={{ marginBottom: 4 }}>{children}</li>,
                      hr: () => <div style={{ height: 1, background: 'rgba(255,255,255,.10)', margin: '12px 0' }} />,
                      strong: ({ children }) => <strong style={{ color: 'rgba(255,255,255,.96)' }}>{children}</strong>,
                      code: ({ children }) => (
                        <code
                          style={{
                            fontFamily: 'var(--mono)',
                            fontSize: 13,
                            background: 'rgba(255,255,255,.08)',
                            padding: '2px 6px',
                            borderRadius: 8,
                            border: '1px solid rgba(255,255,255,.10)',
                          }}
                        >
                          {children}
                        </code>
                      ),
                    }}
                  >
                    {m.content}
                  </ReactMarkdown>
                ) : (
                  <div style={{ whiteSpace: 'pre-wrap' }}>{m.content}</div>
                )}
              </div>
            </div>
          ))}

          {loading ? (
            <div className="row assistant">
              <div className="bubble assistant">
                <div className="typing">
                  <div className="dots">
                    <span />
                    <span />
                    <span />
                  </div>
                  Thinking…
                </div>
              </div>
            </div>
          ) : null}

          <div ref={endRef} />
        </div>

        {/* Composer */}
        <div className="composer">
          <div className="composerInner">
            <textarea
              className="textarea"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about install steps, part lookup, compatibility, or troubleshooting…"
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  sendMessage();
                }
              }}
            />
            <button className="send" onClick={sendMessage} disabled={loading}>
              Send
            </button>
          </div>
          <div className="hint">
            Press <b>Enter</b> to send, <b>Shift+Enter</b> for a new line.
          </div>
        </div>
      </main>
    </div>
  );
}