// ChatPanel — Talk to Lucy
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
    <div className="card fade-in flex flex-col gap-3 max-h-80 overflow-hidden">
      <h3 className="text-xs font-semibold text-[--text2] uppercase tracking-wider">Talk to Lucy</h3>

      <div ref={scrollRef} className="flex-1 overflow-y-auto flex flex-col gap-2.5 min-h-0">
        {chatHistory.length === 0 && (
          <p className="text-sm text-[--text2] italic py-6 text-center">Ask Lucy anything...</p>
        )}
        {chatHistory.map((msg, i) => (
          <div key={i} className={`text-sm px-4 py-3 rounded-2xl ${
            msg.role === 'user'
              ? 'bg-[--blue] text-white ml-12 rounded-br-md'
              : 'bg-[--bg] text-[--text] mr-12 rounded-bl-md'
          }`}>
            {msg.role !== 'user' && <span className="text-xs font-semibold text-[--purple]">Lucy</span>}
            {msg.tick && <span className="text-[--text2] ml-2 text-xs mono">t{msg.tick}</span>}
            <p className={`leading-relaxed ${msg.role === 'user' ? '' : 'mt-0.5'}`}>{msg.text}</p>
          </div>
        ))}
        {loading && (
          <div className="text-sm text-[--purple] py-2 flex items-center gap-2">
            <span className="flex gap-1">
              <span className="w-2 h-2 rounded-full bg-[--purple] animate-pulse" />
              <span className="w-2 h-2 rounded-full bg-[--purple] animate-pulse" style={{ animationDelay: '0.2s' }} />
              <span className="w-2 h-2 rounded-full bg-[--purple] animate-pulse" style={{ animationDelay: '0.4s' }} />
            </span>
            Thinking...
          </div>
        )}
      </div>

      <div className="flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSend()}
          placeholder="Ask Lucy..."
          className="flex-1 bg-[--bg] rounded-xl px-4 py-3 text-sm text-[--text] outline-none focus:ring-2 focus:ring-[--blue]/20 transition-all"
        />
        <button
          onClick={handleSend}
          disabled={loading || !input.trim()}
          className="px-6 py-3 text-sm font-semibold bg-[--blue] text-white rounded-xl hover:opacity-90 disabled:opacity-30 transition-all"
        >
          Send
        </button>
      </div>
    </div>
  );
}
