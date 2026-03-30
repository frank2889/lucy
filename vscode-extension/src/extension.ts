import * as vscode from 'vscode';
import { LucySidebarProvider } from './sidebarProvider';
import * as path from 'path';

let statusBarItem: vscode.StatusBarItem;
let statusInterval: ReturnType<typeof setInterval>;

function getConfig() {
    const config = vscode.workspace.getConfiguration('lucy');
    return {
        host: config.get<string>('host', '209.38.36.51'),
        port: config.get<number>('port', 3000),
        token: config.get<string>('token', ''),
    };
}

function getBaseUrl(): string {
    const { host, port } = getConfig();
    return `http://${host}:${port}`;
}

function authHeaders(): Record<string, string> {
    const { token } = getConfig();
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (token) { headers['Authorization'] = `Bearer ${token}`; }
    return headers;
}

async function checkConnection(): Promise<boolean> {
    try {
        const res = await fetch(`${getBaseUrl()}/status`, { signal: AbortSignal.timeout(3000) });
        return res.ok;
    } catch { return false; }
}

async function updateStatusBar() {
    const connected = await checkConnection();
    if (connected) {
        // Fetch mood
        try {
            const res = await fetch(`${getBaseUrl()}/activity`, { signal: AbortSignal.timeout(3000) });
            if (res.ok) {
                const data = await res.json() as Record<string, unknown>;
                statusBarItem.text = `$(brain) Lucy: ${data.mood}`;
                statusBarItem.color = '#a78bfa';
                statusBarItem.tooltip = `Tick ${(data as any).learning?.tick} | Vocab ${(data as any).learning?.vocabulary_size}`;
                return;
            }
        } catch { /* fallback */ }
        statusBarItem.text = '$(brain) Lucy';
        statusBarItem.color = '#a78bfa';
    } else {
        statusBarItem.text = '$(brain) Lucy (offline)';
        statusBarItem.color = '#ef4444';
    }
}

export function activate(context: vscode.ExtensionContext) {
    const provider = new LucySidebarProvider(context.extensionUri);

    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider('lucy.chatView', provider)
    );

    // Status bar
    statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
    statusBarItem.command = 'lucy.activity';
    statusBarItem.show();
    context.subscriptions.push(statusBarItem);

    updateStatusBar();
    statusInterval = setInterval(updateStatusBar, 10_000);

    // === COMMANDS ===

    // Ask via command palette
    context.subscriptions.push(
        vscode.commands.registerCommand('lucy.ask', async () => {
            const question = await vscode.window.showInputBox({
                prompt: 'Stel Lucy een vraag',
                placeHolder: 'Wat wil je weten?'
            });
            if (question) { provider.sendQuestion(question); }
        })
    );

    // Feed selected text
    context.subscriptions.push(
        vscode.commands.registerCommand('lucy.feedSelection', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) { return vscode.window.showWarningMessage('Geen editor open'); }
            const selection = editor.document.getText(editor.selection);
            if (!selection) { return vscode.window.showWarningMessage('Selecteer eerst tekst'); }

            try {
                const res = await fetch(`${getBaseUrl()}/feed`, {
                    method: 'POST', headers: authHeaders(),
                    body: JSON.stringify({ text: selection }),
                });
                if (res.ok) {
                    vscode.window.showInformationMessage(`Lucy heeft ${selection.length} tekens geleerd`);
                }
            } catch { vscode.window.showErrorMessage('Kan Lucy niet bereiken'); }
        })
    );

    // Feed current file as code
    context.subscriptions.push(
        vscode.commands.registerCommand('lucy.feedFile', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) { return vscode.window.showWarningMessage('Geen bestand open'); }

            const code = editor.document.getText();
            const filename = path.basename(editor.document.fileName);

            const res = await fetch(`${getBaseUrl()}/feed-code`, {
                method: 'POST', headers: authHeaders(),
                body: JSON.stringify({ filename, code }),
            }).catch(() => null);

            if (res?.ok) {
                const data = await res.json() as Record<string, unknown>;
                vscode.window.showInformationMessage(
                    `Lucy leerde van ${filename}: ${data.tokens_trained} tokens, ${data.new_vocab_tokens} nieuwe woorden`
                );
                provider.addSystemMessage(`Geleerd van ${filename}: ${data.tokens_trained} tokens getraind`);
            } else {
                vscode.window.showErrorMessage('Kon bestand niet aan Lucy voeren');
            }
        })
    );

    // Feed all Rust files (Lucy leert haar eigen code)
    context.subscriptions.push(
        vscode.commands.registerCommand('lucy.feedWorkspace', async () => {
            const files = await vscode.workspace.findFiles('crates/**/*.rs', '**/target/**', 50);
            if (files.length === 0) {
                return vscode.window.showWarningMessage('Geen .rs bestanden gevonden');
            }

            const confirm = await vscode.window.showInformationMessage(
                `${files.length} Rust bestanden naar Lucy sturen?`, 'Ja', 'Nee'
            );
            if (confirm !== 'Ja') { return; }

            let total = 0;
            for (const file of files) {
                const doc = await vscode.workspace.openTextDocument(file);
                const code = doc.getText();
                const filename = path.basename(file.fsPath);

                try {
                    const res = await fetch(`${getBaseUrl()}/feed-code`, {
                        method: 'POST', headers: authHeaders(),
                        body: JSON.stringify({ filename, code }),
                    });
                    if (res.ok) { total++; }
                } catch { /* skip failed files */ }
            }

            vscode.window.showInformationMessage(`Lucy leerde van ${total}/${files.length} bestanden`);
            provider.addSystemMessage(`Workspace geleerd: ${total} bestanden verwerkt`);
        })
    );

    // Lucy learn own source code (triggers self-read on server)
    context.subscriptions.push(
        vscode.commands.registerCommand('lucy.learnOwnCode', async () => {
            // Feed Lucy's own crates from the workspace
            await vscode.commands.executeCommand('lucy.feedWorkspace');
        })
    );

    // Activity - what is Lucy doing?
    context.subscriptions.push(
        vscode.commands.registerCommand('lucy.activity', async () => {
            try {
                const res = await fetch(`${getBaseUrl()}/activity`, { signal: AbortSignal.timeout(5000) });
                if (res.ok) {
                    const data = await res.json() as any;
                    provider.showActivity(data);
                }
            } catch { vscode.window.showErrorMessage('Kan Lucy niet bereiken'); }
        })
    );

    // Status
    context.subscriptions.push(
        vscode.commands.registerCommand('lucy.status', async () => {
            try {
                const res = await fetch(`${getBaseUrl()}/status`, { signal: AbortSignal.timeout(3000) });
                if (res.ok) {
                    const data = await res.json() as Record<string, unknown>;
                    vscode.window.showInformationMessage(
                        `Lucy v${data.version} | ${(data as any).field?.points} punten | tick ${data.tick}`
                    );
                }
            } catch { vscode.window.showErrorMessage('Lucy is niet bereikbaar'); }
        })
    );
}

export function deactivate() {
    if (statusInterval) { clearInterval(statusInterval); }
}
