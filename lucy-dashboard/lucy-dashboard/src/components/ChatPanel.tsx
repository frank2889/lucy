import { useState, useRef, useEffect } from 'react';
import { useStore } from '../store';

export default function ChatPanel() {
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const chatHistory = useStore((s) => s.chatHistory);
  const askLucy = useStore((s) => s.askLucy);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [chatHistory]);

  const handleSend = async () => {
    const q = input.trim();
    if (!q || loading) return;
    setInput('');
    setLoading(true);
    try { await askLucy(q); } finally { setLoading(false); }
  };

  return (
    <div className="card" style={{ padding: 20, display: 'flex', flexDirection: 'column', maxHeight: 400 }}>
      <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 16 }}>Talk to Lucy</div>

      <div ref={scrollRef} style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 10, marginBottom: 16 }}>
        {chatHistory.length === 0 && (
          <p style={{ textAlign: 'center', color: 'var(--text-disabled)', padding: '32px 0', fontSize: 13, fontStyle: 'italic' }}>Ask Lucy anything...</p>
        )}
        {chatHistory.map((msg, i) => (
          <div key={i} style={{
            fontSize: 13, padding: '10px 14px', borderRadius: 12, maxWidth: '80%', lineHeight: 1.5,
            ...(msg.role === 'user'
              ? { background: 'var(--blue)', color: '#fff', alignSelf: 'flex-end', borderBottomRightRadius: 4 }
              : { background: 'var(--page-bg)', border: '1px solid var(--border)', alignSelf: 'flex-start', borderBottomLeftRadius: 4 }),
          }}>
            {msg.role !== 'user' && <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--purple)', marginBottom: 2 }}>Lucy</div>}
            {msg.text}
          </div>
        ))}
        {loading && (
          <div style={{ fontSize: 13, color: 'var(--text-disabled)', display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ display: 'flex', gap: 3 }}>
              {[0, 1, 2].map(i => (
                <span key={i} style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--text-disabled)', animation: 'pulse 1.4s infinite', animationDelay: `${i * 0.2}s` }} />
              ))}
            </span>
            Thinking...
          </div>
        )}
      </div>

      <div style={{ display: 'flex', gap: 8 }}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSend()}
          placeholder="Ask Lucy..."
          style={{ flex: 1, padding: '10px 14px', borderRadius: 8, border: '1px solid var(--border)', fontSize: 13, outline: 'none', background: '#fff' }}
        />
        <button
          onClick={handleSend}
          disabled={loading || !input.trim()}
          style={{ padding: '10px 20px', borderRadius: 8, border: 'none', background: 'var(--green)', color: '#fff', fontWeight: 600, fontSize: 13, cursor: 'pointer', opacity: (loading || !input.trim()) ? 0.4 : 1 }}
        >
          Send
        </button>
      </div>
    </div>
  );
}
