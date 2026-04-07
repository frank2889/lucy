import * as vscode from 'vscode';

/**
 * Response van Lucy's /ask endpoint.
 *
 * Bevat het antwoord plus metadata over het denkproces,
 * en een `internal` blok met Lucy's interne staat:
 * emoties, doelen, leervoortgang, interesses, neuromodulatoren.
 *
 * Het `internal` blok wordt in het chatvenster getoond als een
 * uitklapbaar "Waar Lucy mee bezig is" paneel onder elk antwoord.
 */
interface LucyResponse {
    answer: string;
    settle_ticks: number;
    final_tension: number;
    tokens_generated: number;
    think_rounds: number;
    coherence: number;
    facts_retrieved: number;
    internal?: {
        emotions: { curiosity: string; satisfaction: string; frustration: string };
        goals: Array<{ description: string; strategy: string; priority: string; progress: string }>;
        learning: { vocab_size: number; pages_read: number; tokens_processed: number; tick: number };
        interests: string[];
        neuromodulators: { dopamine: string; serotonin: string };
    };
}

/**
 * Sidebar provider voor het Lucy chatvenster in VS Code.
 *
 * Twee tabs:
 * - **Chat** — Stel vragen, zie antwoorden met meta-badges en uitklapbaar intern-status paneel
 * - **Activity** — Live overzicht van stemming, emoties, leren, interesses
 *
 * Communicatie: webview ↔ extension ↔ Lucy API (HTTP op configurable host:port)
 *
 * Het interne-status paneel toont bij elk antwoord:
 * - Actieve doelen (nieuwe goals groen gemarkeerd)
 * - Emoties en neuromodulatoren
 * - Leervoortgang (vocab, pagina's, tokens, tick)
 * - Interesses als tags
 */
export class LucySidebarProvider implements vscode.WebviewViewProvider {
    private _view?: vscode.WebviewView;

    constructor(private readonly _extensionUri: vscode.Uri) {}

    private getBaseUrl(): string {
        const config = vscode.workspace.getConfiguration('lucy');
        return config.get<string>('url', 'https://lucy.world').replace(/\/+$/, '');
    }

    public resolveWebviewView(webviewView: vscode.WebviewView) {
        this._view = webviewView;
        webviewView.webview.options = { enableScripts: true };
        webviewView.webview.html = this._getHtml();

        webviewView.webview.onDidReceiveMessage(async (msg) => {
            if (msg.type === 'ask') { await this._handleQuestion(msg.question); }
            else if (msg.type === 'feed') { await this._handleFeed(msg.text); }
            else if (msg.type === 'get-activity') { await this._fetchActivity(); }
        });
    }

    public async sendQuestion(question: string) {
        if (this._view) {
            this._view.webview.postMessage({ type: 'ask-start', question });
            await this._handleQuestion(question);
        }
    }

    public addSystemMessage(text: string) {
        this._view?.webview.postMessage({ type: 'system', text });
    }

    public showActivity(data: any) {
        this._view?.webview.postMessage({ type: 'activity', data });
    }

    private async _fetchActivity() {
        if (!this._view) return;
        try {
            const res = await fetch(`${this.getBaseUrl()}/activity`, { signal: AbortSignal.timeout(5000) });
            if (res.ok) {
                const data = await res.json();
                this._view.webview.postMessage({ type: 'activity', data });
            }
        } catch {
            this._view.webview.postMessage({ type: 'error', message: 'Kan activiteit niet ophalen' });
        }
    }

    private async _handleQuestion(question: string) {
        if (!this._view) return;
        this._view.webview.postMessage({ type: 'thinking' });

        try {
            const res = await fetch(`${this.getBaseUrl()}/ask`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question }),
            });

            if (!res.ok) {
                this._view.webview.postMessage({ type: 'error', message: `Server error: ${res.status}` });
                return;
            }

            const data = await res.json() as LucyResponse;
            this._view.webview.postMessage({
                type: 'response',
                answer: data.answer,
                meta: {
                    think_rounds: data.think_rounds,
                    coherence: data.coherence,
                    tension: data.final_tension,
                    tokens: data.tokens_generated,
                    ticks: data.settle_ticks,
                    facts: data.facts_retrieved,
                },
                internal: data.internal,
            });
        } catch {
            this._view.webview.postMessage({
                type: 'error',
                message: 'Kan Lucy niet bereiken. Draait ze op ' + this.getBaseUrl() + '?',
            });
        }
    }

    private async _handleFeed(text: string) {
        if (!this._view) return;
        try {
            const res = await fetch(`${this.getBaseUrl()}/feed`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text }),
            });
            this._view.webview.postMessage({ type: 'feed-result', ok: res.ok, chars: text.length });
        } catch {
            this._view.webview.postMessage({ type: 'feed-result', ok: false, chars: 0 });
        }
    }

    private _getHtml(): string {
        return /*html*/`<!DOCTYPE html>
<html lang="nl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
        font-family: var(--vscode-font-family, system-ui);
        font-size: var(--vscode-font-size, 13px);
        color: var(--vscode-foreground);
        background: var(--vscode-panel-background, var(--vscode-sideBar-background));
        display: flex; flex-direction: column; height: 100vh;
    }

    /* Tabs */
    .tabs { display: flex; border-bottom: 1px solid var(--vscode-panel-border, #333); }
    .tab {
        flex: 1; padding: 8px; text-align: center; cursor: pointer;
        font-size: 12px; font-weight: 500; color: var(--vscode-foreground);
        opacity: 0.6; transition: all 0.2s;
        border-bottom: 2px solid transparent;
    }
    .tab:hover { opacity: 0.8; }
    .tab.active { opacity: 1; border-bottom-color: #7c3aed; color: #a78bfa; }

    .panel { display: none; flex: 1; flex-direction: column; overflow: hidden; }
    .panel.active { display: flex; }

    /* Header */
    .header {
        display: flex; align-items: center; gap: 8px;
        padding: 10px 12px; border-bottom: 1px solid var(--vscode-panel-border, #333);
    }
    .header h2 { font-size: 14px; font-weight: 600; color: #a78bfa; }
    .status-dot { width: 8px; height: 8px; border-radius: 50%; background: #ef4444; }
    .status-dot.online { background: #22c55e; }
    .mood-text { font-size: 11px; color: #888; margin-left: auto; }

    /* Chat */
    .messages { flex: 1; overflow-y: auto; padding: 10px; display: flex; flex-direction: column; gap: 10px; }
    .msg { max-width: 92%; padding: 8px 12px; border-radius: 10px; line-height: 1.5; word-wrap: break-word; font-size: 13px; }
    .msg.user { align-self: flex-end; background: #7c3aed; color: #fff; border-bottom-right-radius: 3px; }
    .msg.lucy { align-self: flex-start; background: var(--vscode-editor-background, #1e1e1e); border: 1px solid var(--vscode-panel-border, #333); border-bottom-left-radius: 3px; }
    .msg.error { align-self: center; background: #7f1d1d; color: #fca5a5; font-size: 12px; border-radius: 6px; }
    .msg.system { align-self: center; color: #888; font-size: 11px; font-style: italic; }
    .meta { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 6px; }
    .badge { font-size: 10px; padding: 1px 5px; border-radius: 3px; background: var(--vscode-badge-background, #333); color: var(--vscode-badge-foreground, #aaa); }
    .thinking { align-self: flex-start; color: #a78bfa; font-style: italic; font-size: 12px; padding: 6px 12px; }

    /* Internal state panel */
    .internal-toggle {
        font-size: 11px; color: #7c3aed; cursor: pointer; margin-top: 6px;
        display: flex; align-items: center; gap: 4px; user-select: none;
    }
    .internal-toggle:hover { color: #a78bfa; }
    .internal-panel {
        display: none; margin-top: 8px; padding: 8px; border-radius: 6px;
        background: var(--vscode-editor-background, #111); border: 1px solid #2d1b69;
        font-size: 11px; line-height: 1.6;
    }
    .internal-panel.open { display: block; }
    .internal-section { margin-bottom: 8px; }
    .internal-section:last-child { margin-bottom: 0; }
    .internal-label { color: #a78bfa; font-weight: 600; font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 2px; }
    .goal-item { padding: 3px 0; border-bottom: 1px solid #1a1a2a; }
    .goal-item:last-child { border: none; }
    .goal-desc { color: var(--vscode-foreground); }
    .goal-strategy { color: #888; font-size: 10px; }
    .goal-progress { color: #a78bfa; font-size: 10px; float: right; }
    .emotion-row { display: flex; justify-content: space-between; padding: 1px 0; }
    .emotion-name { color: #888; }
    .emotion-val { font-weight: 500; }
    .interest-chip {
        display: inline-block; font-size: 10px; padding: 1px 5px; margin: 1px;
        border-radius: 3px; background: #2d1b69; color: #c4b5fd;
    }
    .learn-row { display: flex; justify-content: space-between; padding: 1px 0; color: #888; }
    .learn-val { color: var(--vscode-foreground); font-weight: 500; }

    /* Input */
    .input-area { display: flex; gap: 6px; padding: 10px; border-top: 1px solid var(--vscode-panel-border, #333); }
    .input-area input {
        flex: 1; padding: 7px 10px; border: 1px solid var(--vscode-input-border, #444);
        border-radius: 6px; background: var(--vscode-input-background, #1e1e1e);
        color: var(--vscode-input-foreground, #fff); font-size: 13px; outline: none;
    }
    .input-area input:focus { border-color: #7c3aed; }
    .input-area button {
        padding: 7px 14px; border: none; border-radius: 6px;
        background: #7c3aed; color: #fff; cursor: pointer; font-size: 12px; font-weight: 500;
    }
    .input-area button:hover { background: #6d28d9; }
    .input-area button:disabled { opacity: 0.5; cursor: not-allowed; }

    /* Activity panel */
    .activity { flex: 1; overflow-y: auto; padding: 12px; }
    .activity h3 { font-size: 12px; font-weight: 600; color: #a78bfa; margin: 12px 0 6px 0; text-transform: uppercase; letter-spacing: 0.5px; }
    .activity h3:first-child { margin-top: 0; }
    .stat { display: flex; justify-content: space-between; padding: 3px 0; font-size: 12px; }
    .stat .label { color: #888; }
    .stat .value { color: var(--vscode-foreground); font-weight: 500; }
    .emotion-bar { height: 4px; border-radius: 2px; background: #333; margin: 2px 0 6px 0; }
    .emotion-fill { height: 100%; border-radius: 2px; transition: width 0.5s; }
    .interest-tag {
        display: inline-block; font-size: 11px; padding: 2px 6px; margin: 2px;
        border-radius: 3px; background: #2d1b69; color: #c4b5fd;
    }
    .observation { font-size: 11px; color: #999; padding: 4px 0; border-bottom: 1px solid #222; }
    .observation .time { color: #666; }
    .refresh-btn {
        width: 100%; padding: 8px; margin-top: 10px; border: 1px solid #444;
        border-radius: 6px; background: transparent; color: #a78bfa;
        cursor: pointer; font-size: 12px;
    }
    .refresh-btn:hover { background: #1a1a2e; }
</style>
</head>
<body>
    <div class="tabs">
        <div class="tab active" data-tab="chat">Chat</div>
        <div class="tab" data-tab="activity">Activiteit</div>
    </div>

    <!-- CHAT TAB -->
    <div class="panel active" id="panel-chat">
        <div class="header">
            <div class="status-dot" id="statusDot"></div>
            <h2>Lucy</h2>
            <span class="mood-text" id="moodText"></span>
        </div>
        <div class="messages" id="messages"></div>
        <div class="input-area">
            <input type="text" id="input" placeholder="Stel Lucy een vraag..." />
            <button id="sendBtn">Vraag</button>
        </div>
    </div>

    <!-- ACTIVITY TAB -->
    <div class="panel" id="panel-activity">
        <div class="activity" id="activityContent">
            <p style="color:#888; text-align:center; margin-top:40px;">Laden...</p>
        </div>
        <button class="refresh-btn" id="refreshBtn">Ververs</button>
    </div>

<script>
    const vscode = acquireVsCodeApi();
    const messages = document.getElementById('messages');
    const input = document.getElementById('input');
    const sendBtn = document.getElementById('sendBtn');
    const statusDot = document.getElementById('statusDot');
    const moodText = document.getElementById('moodText');
    const activityContent = document.getElementById('activityContent');
    let thinking = false;

    // Tabs
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
            tab.classList.add('active');
            document.getElementById('panel-' + tab.dataset.tab).classList.add('active');
            if (tab.dataset.tab === 'activity') {
                vscode.postMessage({ type: 'get-activity' });
            }
        });
    });

    document.getElementById('refreshBtn').addEventListener('click', () => {
        vscode.postMessage({ type: 'get-activity' });
    });

    function esc(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

    function addMessage(html, cls) {
        const el = messages.querySelector('.thinking');
        if (el) el.remove();
        const div = document.createElement('div');
        div.className = 'msg ' + cls;
        div.innerHTML = html;
        messages.appendChild(div);
        messages.scrollTop = messages.scrollHeight;
    }

    function showThinking() {
        if (messages.querySelector('.thinking')) return;
        const div = document.createElement('div');
        div.className = 'thinking';
        div.textContent = 'Lucy denkt na...';
        messages.appendChild(div);
        messages.scrollTop = messages.scrollHeight;
    }

    function send() {
        const q = input.value.trim();
        if (!q || thinking) return;
        thinking = true;
        sendBtn.disabled = true;
        addMessage(esc(q), 'user');
        input.value = '';
        vscode.postMessage({ type: 'ask', question: q });
    }

    input.addEventListener('keydown', e => { if (e.key === 'Enter') send(); });
    sendBtn.addEventListener('click', send);

    function renderActivity(data) {
        let html = '';

        // Mood & Emotions
        html += '<h3>Stemming: ' + esc(data.mood || '?') + '</h3>';
        const emotions = data.emotions || {};
        for (const [name, val] of Object.entries(emotions)) {
            const pct = Math.min(100, Math.max(0, parseFloat(val) * 100));
            const color = name === 'curiosity' ? '#818cf8' : name === 'satisfaction' ? '#34d399' : '#f87171';
            html += '<div class="stat"><span class="label">' + name + '</span><span class="value">' + val + '</span></div>';
            html += '<div class="emotion-bar"><div class="emotion-fill" style="width:' + pct + '%;background:' + color + '"></div></div>';
        }

        // Neuromodulators
        html += '<h3>Neurochemie</h3>';
        const neuro = data.neuromodulators || {};
        for (const [name, val] of Object.entries(neuro)) {
            html += '<div class="stat"><span class="label">' + name + '</span><span class="value">' + val + '</span></div>';
        }

        // Learning
        html += '<h3>Leren</h3>';
        const learn = data.learning || {};
        html += '<div class="stat"><span class="label">Tick</span><span class="value">' + (learn.tick || 0).toLocaleString() + '</span></div>';
        html += '<div class="stat"><span class="label">Vocabulaire</span><span class="value">' + (learn.vocabulary_size || 0).toLocaleString() + '</span></div>';
        html += '<div class="stat"><span class="label">Spanning</span><span class="value">' + (learn.avg_tension || '?') + '</span></div>';
        html += '<div class="stat"><span class="label">Fase</span><span class="value">' + (learn.phase || '?') + '</span></div>';

        // Exploring
        html += '<h3>Exploratie</h3>';
        const exp = data.exploring || {};
        html += '<div class="stat"><span class="label">Paginas gelezen</span><span class="value">' + (exp.pages_read || 0) + '</span></div>';
        html += '<div class="stat"><span class="label">Tokens verwerkt</span><span class="value">' + (exp.tokens_processed || 0).toLocaleString() + '</span></div>';

        // Interests
        const interests = exp.current_interests || [];
        if (interests.length > 0) {
            html += '<h3>Interesses</h3>';
            html += '<div>';
            for (const i of interests) {
                html += '<span class="interest-tag">' + esc(i) + '</span>';
            }
            html += '</div>';
        }

        // Recent observations
        const obs = data.recent_observations || [];
        if (obs.length > 0) {
            html += '<h3>Recente observaties</h3>';
            for (const o of obs.slice(0, 8)) {
                html += '<div class="observation">' + esc(o.description || o.event || JSON.stringify(o)) + '</div>';
            }
        }

        activityContent.innerHTML = html;
    }

    window.addEventListener('message', event => {
        const msg = event.data;
        switch (msg.type) {
            case 'thinking': showThinking(); break;
            case 'response': {
                thinking = false;
                sendBtn.disabled = false;
                let html = esc(msg.answer || '(geen antwoord)');
                if (msg.meta) {
                    html += '<div class="meta">';
                    html += '<span class="badge">' + msg.meta.think_rounds + ' rondes</span>';
                    html += '<span class="badge">' + (msg.meta.coherence * 100).toFixed(0) + '% coherent</span>';
                    if (msg.meta.facts > 0) html += '<span class="badge">' + msg.meta.facts + ' feiten</span>';
                    html += '</div>';
                }
                if (msg.internal) {
                    const uid = 'int-' + Date.now();
                    html += '<div class="internal-toggle" onclick="document.getElementById(\'' + uid + '\').classList.toggle(\'open\')">';
                    html += '<span>&#9662;</span> Waar Lucy mee bezig is';
                    html += '</div>';
                    html += '<div class="internal-panel" id="' + uid + '">';

                    // Doelen
                    const goals = msg.internal.goals || [];
                    if (goals.length > 0) {
                        html += '<div class="internal-section"><div class="internal-label">Doelen</div>';
                        for (const g of goals) {
                            html += '<div class="goal-item">';
                            if (g.new) html += '<span style="color:#22c55e;font-size:10px">NIEUW </span>';
                            html += '<span class="goal-desc">' + esc(g.description) + '</span>';
                            html += '<span class="goal-progress">' + esc(g.progress) + '</span>';
                            html += '<br><span class="goal-strategy">' + esc(g.strategy) + '</span>';
                            html += '</div>';
                        }
                        html += '</div>';
                    } else {
                        html += '<div class="internal-section"><div class="internal-label">Doelen</div><span style="color:#666">Nog geen actieve doelen</span></div>';
                    }

                    // Emoties
                    const emo = msg.internal.emotions || {};
                    html += '<div class="internal-section"><div class="internal-label">Emoties</div>';
                    const emoNames = { curiosity: 'Nieuwsgierigheid', satisfaction: 'Tevredenheid', frustration: 'Frustratie' };
                    for (const [k, label] of Object.entries(emoNames)) {
                        const v = emo[k] || '0';
                        html += '<div class="emotion-row"><span class="emotion-name">' + label + '</span><span class="emotion-val">' + v + '</span></div>';
                    }
                    const neuro = msg.internal.neuromodulators || {};
                    html += '<div class="emotion-row" style="margin-top:4px;border-top:1px solid #1a1a2a;padding-top:4px"><span class="emotion-name">Dopamine</span><span class="emotion-val">' + (neuro.dopamine || '0') + '</span></div>';
                    html += '<div class="emotion-row"><span class="emotion-name">Serotonine</span><span class="emotion-val">' + (neuro.serotonin || '0') + '</span></div>';
                    html += '</div>';

                    // Leervoortgang
                    const learn = msg.internal.learning || {};
                    html += '<div class="internal-section"><div class="internal-label">Leervoortgang</div>';
                    html += '<div class="learn-row"><span>Vocabulaire</span><span class="learn-val">' + (learn.vocab_size || 0).toLocaleString() + ' woorden</span></div>';
                    html += '<div class="learn-row"><span>Paginas gelezen</span><span class="learn-val">' + (learn.pages_read || 0) + '</span></div>';
                    html += '<div class="learn-row"><span>Tokens verwerkt</span><span class="learn-val">' + (learn.tokens_processed || 0).toLocaleString() + '</span></div>';
                    html += '<div class="learn-row"><span>Huidige tick</span><span class="learn-val">' + (learn.tick || 0).toLocaleString() + '</span></div>';
                    html += '</div>';

                    // Interesses
                    const interests = msg.internal.interests || [];
                    if (interests.length > 0) {
                        html += '<div class="internal-section"><div class="internal-label">Interesses</div>';
                        for (const i of interests) {
                            html += '<span class="interest-chip">' + esc(i) + '</span>';
                        }
                        html += '</div>';
                    }

                    html += '</div>';
                }
                addMessage(html, 'lucy');
                break;
            }
            case 'error':
                thinking = false; sendBtn.disabled = false;
                addMessage(esc(msg.message), 'error');
                break;
            case 'ask-start':
                addMessage(esc(msg.question), 'user');
                showThinking();
                break;
            case 'system':
                addMessage(esc(msg.text), 'system');
                break;
            case 'feed-result':
                if (msg.ok) addMessage('Geleerd: ' + msg.chars + ' tekens', 'system');
                else addMessage('Kon niet aan Lucy voeren', 'error');
                break;
            case 'activity':
                renderActivity(msg.data);
                moodText.textContent = msg.data.mood || '';
                statusDot.classList.add('online');
                break;
        }
    });

    statusDot.classList.add('online');
    // Auto-fetch activity on load
    setTimeout(() => vscode.postMessage({ type: 'get-activity' }), 1000);
</script>
</body>
</html>`;
    }
}
