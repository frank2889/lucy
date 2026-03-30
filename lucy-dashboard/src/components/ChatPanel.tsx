// ChatPanel — Talk to Lucy via /ask endpoint
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
    try {
      await askLucy(q);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="glass p-3 flex flex-col gap-2 max-h-full overflow-hidden">
      <h3 className="text-xs uppercase tracking-widest text-[--text-secondary]">Talk to Lucy</h3>

      <div ref={scrollRef} className="flex-1 overflow-y-auto flex flex-col gap-2 min-h-0">
        {chatHistory.length === 0 && (
          <p className="text-xs text-[--text-secondary] italic">Ask Lucy anything...</p>
        )}
        {chatHistory.map((msg, i) => (
          <div
            key={i}
            className={`text-xs px-2 py-1.5 rounded ${
              msg.role === 'user'
                ? 'bg-[--accent-blue]/10 border border-[--accent-blue]/20 ml-4'
                : 'bg-[--bg-secondary] border border-[--border] mr-4'
            }`}
          >
            <span className={`font-bold ${msg.role === 'user' ? 'text-[--accent-blue]' : 'text-[--accent-cyan]'}`}>
              {msg.role === 'user' ? 'You' : 'Lucy'}
            </span>
            {msg.tick && <span className="text-[--text-secondary] ml-2">t{msg.tick}</span>}
            <p className="mt-0.5 leading-relaxed whitespace-pre-wrap">{msg.text}</p>
          </div>
        ))}
        {loading && (
          <div className="text-xs text-[--accent-cyan] animate-pulse-slow">Lucy is thinking...</div>
        )}
      </div>

      <div className="flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSend()}
          placeholder="Ask Lucy..."
          className="flex-1 bg-[--bg-secondary] border border-[--border] rounded px-2 py-1.5 text-xs text-[--text-primary] outline-none focus:border-[--accent-blue] transition-colors"
        />
        <button
          onClick={handleSend}
          disabled={loading || !input.trim()}
          className="px-3 py-1.5 text-xs bg-[--accent-blue]/20 border border-[--accent-blue]/40 rounded text-[--accent-blue] hover:bg-[--accent-blue]/30 disabled:opacity-40 transition-colors"
        >
          Send
        </button>
      </div>
    </div>
  );
}
