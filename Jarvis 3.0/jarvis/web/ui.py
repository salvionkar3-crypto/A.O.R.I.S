"""Embedded Desktop Web Interface for A.O.R.I.S."""

JARVIS_HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>A.O.R.I.S. — Multi-Model AI Orchestration System | APEX LEGENDS</title>
    <style>
        :root {
            --bg-base: #0a0d14;
            --bg-surface: #121824;
            --bg-card: #182030;
            --border-color: #223048;
            --accent-cyan: #00e5ff;
            --accent-cyan-dim: rgba(0, 229, 255, 0.12);
            --accent-blue: #0077ff;
            --text-main: #e2e8f0;
            --text-muted: #8899b0;
            --status-success: #00e676;
            --status-warning: #ffab00;
            --status-danger: #ff3d71;
            --font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            --font-mono: "Fira Code", Consolas, Monaco, "Courier New", monospace;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background-color: var(--bg-base);
            color: var(--text-main);
            font-family: var(--font-family);
            display: flex;
            flex-direction: column;
            height: 100vh;
            overflow: hidden;
        }

        /* Top Bar */
        header {
            background-color: var(--bg-surface);
            border-bottom: 1px solid var(--border-color);
            padding: 12px 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .arc-reactor {
            width: 28px;
            height: 28px;
            border-radius: 50%;
            border: 2px solid var(--accent-cyan);
            box-shadow: 0 0 12px var(--accent-cyan);
            display: flex;
            align-items: center;
            justify-content: center;
            animation: pulse 3s infinite ease-in-out;
        }
        .arc-inner {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background-color: var(--accent-cyan);
        }
        @keyframes pulse {
            0%, 100% { box-shadow: 0 0 8px var(--accent-cyan); }
            50% { box-shadow: 0 0 20px var(--accent-cyan); }
        }
        .brand-title {
            font-size: 1.25rem;
            font-weight: 700;
            letter-spacing: 2px;
            color: #ffffff;
        }
        .brand-sub {
            font-size: 0.75rem;
            color: var(--accent-cyan);
            letter-spacing: 1px;
            text-transform: uppercase;
        }
        .nav-tabs {
            display: flex;
            gap: 8px;
        }
        .nav-tab {
            background: none;
            border: 1px solid transparent;
            color: var(--text-muted);
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 0.85rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        .nav-tab:hover {
            color: var(--text-main);
            background-color: var(--bg-card);
        }
        .nav-tab.active {
            color: var(--accent-cyan);
            background-color: var(--accent-cyan-dim);
            border-color: rgba(0, 229, 255, 0.4);
        }
        .header-status {
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 0.8rem;
        }
        .badge {
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
        }
        .badge-success { background: rgba(0, 230, 118, 0.15); color: var(--status-success); border: 1px solid rgba(0, 230, 118, 0.3); }
        .badge-warning { background: rgba(255, 171, 0, 0.15); color: var(--status-warning); border: 1px solid rgba(255, 171, 0, 0.3); }
        .badge-danger { background: rgba(255, 61, 113, 0.15); color: var(--status-danger); border: 1px solid rgba(255, 61, 113, 0.3); }
        .badge-cyan { background: var(--accent-cyan-dim); color: var(--accent-cyan); border: 1px solid rgba(0, 229, 255, 0.3); }

        /* Main Workspace */
        main {
            flex: 1;
            display: flex;
            overflow: hidden;
            position: relative;
        }

        .tab-content {
            flex: 1;
            display: none;
            height: 100%;
            overflow-y: auto;
        }
        .tab-content.active {
            display: flex;
        }

        /* 1. Chat Tab */
        #tab-chat {
            flex-direction: row;
        }
        .chat-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            border-right: 1px solid var(--border-color);
            background-color: var(--bg-base);
        }
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .message-row {
            display: flex;
            flex-direction: column;
            max-width: 80%;
            gap: 6px;
        }
        .message-row.user {
            align-self: flex-end;
        }
        .message-row.assistant {
            align-self: flex-start;
        }
        .message-header {
            font-size: 0.75rem;
            color: var(--text-muted);
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .message-bubble {
            padding: 14px 18px;
            border-radius: 8px;
            font-size: 0.95rem;
            line-height: 1.5;
            white-space: pre-wrap;
            word-break: break-word;
        }
        .user .message-bubble {
            background-color: #1e3a5f;
            color: #ffffff;
            border: 1px solid #2b5488;
        }
        .assistant .message-bubble {
            background-color: var(--bg-surface);
            color: var(--text-main);
            border: 1px solid var(--border-color);
        }
        .verification-box {
            margin-top: 8px;
            padding: 10px 14px;
            background-color: rgba(18, 24, 36, 0.8);
            border-left: 3px solid var(--accent-cyan);
            border-radius: 4px;
            font-size: 0.8rem;
        }
        .chat-input-bar {
            padding: 16px 24px;
            background-color: var(--bg-surface);
            border-top: 1px solid var(--border-color);
            display: flex;
            gap: 12px;
        }
        .chat-input-bar input {
            flex: 1;
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            color: var(--text-main);
            padding: 12px 16px;
            border-radius: 6px;
            font-size: 0.95rem;
            outline: none;
            transition: border-color 0.2s;
        }
        .chat-input-bar input:focus {
            border-color: var(--accent-cyan);
        }
        .btn {
            background-color: var(--accent-cyan);
            color: #050a12;
            border: none;
            padding: 12px 24px;
            border-radius: 6px;
            font-weight: 700;
            font-size: 0.9rem;
            cursor: pointer;
            transition: all 0.2s;
        }
        .btn:hover {
            box-shadow: 0 0 12px var(--accent-cyan);
        }
        .btn-danger {
            background-color: var(--status-danger);
            color: #ffffff;
        }
        .btn-secondary {
            background-color: var(--bg-card);
            color: var(--text-main);
            border: 1px solid var(--border-color);
        }
        .btn-secondary:hover {
            background-color: #223048;
        }

        /* Confirmation Alert Banner */
        .confirmation-banner {
            background-color: rgba(255, 171, 0, 0.1);
            border: 1px solid var(--status-warning);
            padding: 14px 20px;
            margin: 12px 24px;
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
        }

        /* Side Info Panel */
        .side-panel {
            width: 340px;
            background-color: var(--bg-surface);
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 20px;
            overflow-y: auto;
        }
        .side-title {
            font-size: 0.85rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-muted);
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 8px;
        }
        .memory-chip {
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            padding: 10px;
            border-radius: 6px;
            font-size: 0.8rem;
            margin-bottom: 8px;
        }

        /* 2. Memory Tab */
        #tab-memories {
            padding: 24px;
            flex-direction: column;
            gap: 20px;
        }
        .controls-row {
            display: flex;
            gap: 12px;
            align-items: center;
        }
        .search-box {
            flex: 1;
            padding: 10px 14px;
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            color: var(--text-main);
            border-radius: 6px;
        }
        .memory-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 16px;
            overflow-y: auto;
        }
        .memory-card {
            background-color: var(--bg-surface);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .card-body {
            font-size: 0.9rem;
            color: var(--text-main);
            flex: 1;
        }
        .card-footer {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.75rem;
            color: var(--text-muted);
            border-top: 1px solid var(--border-color);
            padding-top: 8px;
        }

        /* 3. Tasks Tab */
        #tab-tasks {
            padding: 24px;
            flex-direction: column;
            gap: 16px;
        }
        .task-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
            background: var(--bg-surface);
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid var(--border-color);
        }
        .task-table th, .task-table td {
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }
        .task-table th {
            background-color: var(--bg-card);
            color: var(--text-muted);
            font-weight: 600;
        }

        /* 4. Settings Tab */
        #tab-settings {
            padding: 32px;
            flex-direction: column;
            gap: 24px;
            max-width: 800px;
        }
        .settings-card {
            background: var(--bg-surface);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .form-group {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }
        .form-group label {
            font-size: 0.85rem;
            color: var(--text-muted);
        }
        .form-group input {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            color: var(--text-main);
            padding: 10px;
            border-radius: 6px;
        }
    </style>
</head>
<body>

    <header>
        <div class="brand">
            <div class="arc-reactor"><div class="arc-inner"></div></div>
            <div>
                <div class="brand-title">A.O.R.I.S.</div>
                <div class="brand-sub">Multi-Model AI Orchestrator</div>
            </div>
        </div>

        <nav class="nav-tabs">
            <button class="nav-tab active" onclick="switchTab('chat')">💬 Live Dialogue</button>
            <button class="nav-tab" onclick="switchTab('memories')">🧠 Memory Matrix</button>
            <button class="nav-tab" onclick="switchTab('tasks')">⚡ Task Ledger</button>
            <button class="nav-tab" onclick="switchTab('settings')">⚙️ System & Config</button>
        </nav>

        <!-- Multi-Model Operational Status Bar -->
        <div id="model-status-bar" style="display: flex; gap: 6px; align-items: center;">
            <span id="pill-nemotron" class="badge" style="background: rgba(0, 229, 255, 0.1); border: 1px solid var(--accent-cyan); color: var(--accent-cyan); cursor: default;" title="Primary Orchestrator & Final Synthesis">Nemotron</span>
            <span id="pill-openai" class="badge" style="background: rgba(255, 255, 255, 0.05); border: 1px solid var(--border-color); color: var(--text-muted); cursor: default;" title="Reasoning & Deep Analysis">OpenAI</span>
            <span id="pill-anthropic" class="badge" style="background: rgba(255, 255, 255, 0.05); border: 1px solid var(--border-color); color: var(--text-muted); cursor: default;" title="Code & Technical Engineering">Anthropic</span>
            <span id="pill-google" class="badge" style="background: rgba(255, 255, 255, 0.05); border: 1px solid var(--border-color); color: var(--text-muted); cursor: default;" title="Multimodal Vision & Creative">Google</span>
        </div>

        <div class="header-status">
            <button id="auth-btn" class="btn" style="padding: 4px 10px; font-size: 0.75rem; background: var(--bg-card); border: 1px solid var(--border-color); color: var(--text-muted);" onclick="promptApiKey()">🔑 Access Key</button>
            <span class="badge badge-cyan" id="provider-badge">Provider: Initializing...</span>
            <span class="badge badge-success" id="status-badge">SYSTEM READY</span>
        </div>
    </header>

    <div id="offline-alert-container"></div>

    <main>
        <!-- TAB 1: CHAT & WORKFLOW -->
        <div id="tab-chat" class="tab-content active">
            <div class="chat-container">
                <div id="confirmation-area"></div>
                <div class="chat-messages" id="chat-messages">
                    <div class="message-row assistant">
                        <div class="message-header">A.O.R.I.S. SYSTEM &bull; Just now</div>
                        <div class="message-bubble">Greetings. I am A.O.R.I.S., a multi-model AI assistant orchestrated by NVIDIA Nemotron with specialist support from OpenAI, Anthropic, and Google. How may I assist you today?</div>
                    </div>
                </div>
                <div class="chat-input-bar" style="display: flex; gap: 8px; align-items: center;">
                    <select id="specialist-select" style="background: var(--bg-card); border: 1px solid var(--border-color); color: var(--accent-cyan); padding: 10px; border-radius: 6px; font-size: 0.8rem; cursor: pointer; outline: none;">
                        <option value="">Auto (Nemotron Routing)</option>
                        <option value="openai">OpenAI (Reasoning)</option>
                        <option value="anthropic">Anthropic (Coding)</option>
                        <option value="google">Google (Vision)</option>
                    </select>
                    <input type="text" id="user-input" placeholder="Type a request, state a preference, or assign a task..." onkeydown="if(event.key==='Enter') sendMessage()" style="flex: 1;">
                    <button class="btn" onclick="sendMessage()">Transmit</button>
                </div>
            </div>

            <div class="side-panel">
                <div class="side-title">Active Memories in Context</div>
                <div id="retrieved-memories-container">
                    <div style="font-size: 0.8rem; color: var(--text-muted);">No memories currently loaded in active context.</div>
                </div>
                <div class="side-title">Active Orchestrator State</div>
                <div id="orchestrator-state-box" style="font-size: 0.8rem; color: var(--text-muted);">
                    Status: IDLE<br>Session: default_session
                </div>
            </div>
        </div>

        <!-- TAB 2: PERSISTENT MEMORY ENGINE -->
        <div id="tab-memories" class="tab-content">
            <div class="controls-row">
                <input type="text" id="memory-search-input" class="search-box" placeholder="Search persistent memory via FTS5 BM25 keyword index..." oninput="searchMemories()">
                <button class="btn" onclick="openAddMemoryModal()">+ Save Memory</button>
            </div>
            <div class="memory-grid" id="memory-cards-container">
                <!-- Dynamically loaded memory cards -->
            </div>
        </div>

        <!-- TAB 3: TASK LEDGER -->
        <div id="tab-tasks" class="tab-content">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h2>Task Ledger & Empirical Verification Records</h2>
                <button class="btn btn-secondary" onclick="loadTasks()">Refresh Ledger</button>
            </div>
            <table class="task-table">
                <thead>
                    <tr>
                        <th>Task ID</th>
                        <th>User Request</th>
                        <th>Status</th>
                        <th>Verdict</th>
                        <th>Created At</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody id="task-table-body">
                    <!-- Dynamically populated task rows -->
                </tbody>
            </table>
        </div>

        <!-- TAB 4: SYSTEM & SETTINGS -->
        <div id="tab-settings" class="tab-content">
            <h2>System Health & Provider Configuration</h2>
            <div class="settings-card">
                <h3>Model Provider Status</h3>
                <div id="provider-health-details" style="font-size: 0.9rem; line-height: 1.6;">Checking connectivity...</div>
                <button class="btn btn-secondary" style="width: 180px;" onclick="checkProviderHealth()">Ping Endpoint</button>
            </div>

            <div class="settings-card">
                <h3>Active Configuration</h3>
                <div id="active-config-details" style="font-size: 0.85rem; font-family: var(--font-mono); white-space: pre-wrap; background: var(--bg-card); padding: 12px; border-radius: 6px;">Loading settings...</div>
            </div>
        </div>
    </main>

    <script>
        const SESSION_ID = "session_" + Math.random().toString(36).substring(2, 9);

        function getAuthHeaders(headers = {}) {
            const key = localStorage.getItem('jarvis_api_key');
            if (key) {
                headers['X-API-Key'] = key;
            }
            return headers;
        }

        async function secureFetch(url, options = {}) {
            options.headers = getAuthHeaders(options.headers || {});
            try {
                const res = await fetch(url, options);
                if (res.status === 401) {
                    const promptKey = confirm("This JARVIS instance requires an Access Key (HTTP 401 Unauthorized).\\nWould you like to enter your access key now?");
                    if (promptKey) {
                        promptApiKey();
                    }
                }
                return res;
            } catch (err) {
                console.error("Fetch failed:", err);
                throw err;
            }
        }

        function promptApiKey() {
            const current = localStorage.getItem('jarvis_api_key') || "";
            const key = prompt("Enter your JARVIS Access Key (configured in JARVIS_API_KEY):", current);
            if (key !== null) {
                if (key.trim()) {
                    localStorage.setItem('jarvis_api_key', key.trim());
                } else {
                    localStorage.removeItem('jarvis_api_key');
                }
                updateAuthBadge();
                checkProviderHealth();
            }
        }

        function updateAuthBadge() {
            const btn = document.getElementById('auth-btn');
            if (!btn) return;
            const hasKey = !!localStorage.getItem('jarvis_api_key');
            btn.innerHTML = hasKey ? "🔒 Key Set" : "🔑 Access Key";
            btn.style.color = hasKey ? "var(--status-success)" : "var(--text-muted)";
            btn.style.borderColor = hasKey ? "rgba(0, 230, 118, 0.4)" : "var(--border-color)";
        }

        function switchTab(tabId) {
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.nav-tab').forEach(el => el.classList.remove('active'));
            document.getElementById('tab-' + tabId).classList.add('active');
            event.target.classList.add('active');

            if (tabId === 'memories') loadMemories();
            if (tabId === 'tasks') loadTasks();
            if (tabId === 'settings') loadSettings();
        }

        async function sendMessage() {
            const input = document.getElementById('user-input');
            const sendBtn = document.querySelector('.chat-input-box button');
            const text = input.value.trim();
            if (!text) return;
            input.value = '';
            input.disabled = true;
            if (sendBtn) {
                sendBtn.disabled = true;
                sendBtn.innerText = "Running...";
            }

            const chatMessages = document.getElementById('chat-messages');
            chatMessages.innerHTML += `
                <div class="message-row user">
                    <div class="message-header">YOU &bull; Just now</div>
                    <div class="message-bubble">${escapeHtml(text)}</div>
                </div>
            `;
            chatMessages.scrollTop = chatMessages.scrollHeight;

            document.getElementById('status-badge').className = "badge badge-warning";
            document.getElementById('status-badge').innerText = "PROCESSING...";

            try {
                const specEl = document.getElementById('specialist-select');
                const specVal = specEl ? specEl.value : null;
                const res = await secureFetch('/api/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        message: text,
                        session_id: SESSION_ID,
                        specialist_override: specVal || null
                    })
                });
                const data = await res.json();

                renderTaskResponse(data);
                checkMultiModelStatus();
            } catch (err) {
                chatMessages.innerHTML += `
                    <div class="message-row assistant">
                        <div class="message-header">A.O.R.I.S. &bull; Error</div>
                        <div class="message-bubble" style="border-color: var(--status-danger); color: var(--status-danger);">Error communicating with A.O.R.I.S. backend: ${err.message}</div>
                    </div>
                `;
            } finally {
                input.disabled = false;
                if (sendBtn) {
                    sendBtn.disabled = false;
                    sendBtn.innerText = "Transmit";
                }
                input.focus();
                document.getElementById('status-badge').className = "badge badge-success";
                document.getElementById('status-badge').innerText = "SYSTEM READY";
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
        }

        function renderTaskResponse(data) {
            const chatMessages = document.getElementById('chat-messages');
            const task = data.task;

            let verificationHtml = '';
            if (task.verification) {
                const v = task.verification;
                const vColor = v.verdict === 'verified' ? 'var(--status-success)' : (v.verdict === 'incomplete' ? 'var(--status-warning)' : 'var(--status-danger)');
                verificationHtml = `
                    <div class="verification-box">
                        <div style="font-weight: 700; color: ${vColor};">VERIFICATION: ${v.verdict.toUpperCase()}</div>
                        <div style="color: var(--text-muted); font-size: 0.75rem; margin-top: 2px;">${escapeHtml(v.evidence)}</div>
                    </div>
                `;
            }

            // Mode badge (Live inference vs Offline fallback)
            const modeBadge = data.is_live
                ? `<span class="badge badge-success">🌐 LIVE: ${escapeHtml(data.model || "NVIDIA Nemotron")}</span>`
                : `<span class="badge badge-warning">⚠️ OFFLINE: ${escapeHtml(data.provider_name || "offline")}</span>`;

            // Multi-model badges
            const orchBadge = `<span class="badge" style="border-color: var(--accent-cyan); color: var(--accent-cyan);">🧠 ORCHESTRATOR: ${escapeHtml(data.orchestrator || "NVIDIA Nemotron")}</span>`;
            const specialistBadge = data.specialist_used
                ? `<span class="badge" style="border-color: #a855f7; color: #c084fc;">⚡ SPECIALIST: ${escapeHtml(data.specialist_used)}</span>`
                : '';
            const synthBadge = data.synthesis_performed
                ? `<span class="badge" style="border-color: var(--status-success); color: var(--status-success);">✨ SYNTHESIZED</span>`
                : '';

            // Document Download Box
            let downloadHtml = '';
            if (data.download_url) {
                downloadHtml = `
                    <div style="margin-top: 10px; padding: 12px; background: rgba(0, 229, 255, 0.08); border: 1px solid var(--accent-cyan); border-radius: 6px; display: flex; align-items: center; justify-content: space-between;">
                        <div>
                            <div style="font-size: 0.85rem; font-weight: 700; color: #ffffff;">📄 Generated Document</div>
                            <div style="font-size: 0.75rem; color: var(--text-muted);">${escapeHtml(data.file_name || 'report.pdf')}</div>
                        </div>
                        <a href="${escapeHtml(data.download_url)}" target="_blank" download class="btn" style="padding: 6px 14px; font-size: 0.8rem; text-decoration: none; display: inline-flex; align-items: center; gap: 4px;">📥 Download PDF</a>
                    </div>
                `;
            }

            // Workflow badge
            const wf = task.workflow_type || "conversational";
            const wfLabel = wf.replace(/_/g, ' ').toUpperCase();
            const wfBadge = `<span class="badge badge-cyan">${escapeHtml(wfLabel)}</span>`;

            // Complexity badge
            const cx = (task.complexity || "low").toUpperCase();
            const cxBadgeColor = cx === 'HIGH' ? 'var(--status-danger)' : (cx === 'MEDIUM' ? 'var(--status-warning)' : 'var(--status-success)');
            const cxBadge = `<span class="badge" style="border-color: ${cxBadgeColor}; color: ${cxBadgeColor};">${cx} COMPLEXITY</span>`;

            // Plan steps progress
            let planHtml = '';
            if (task.plan && task.plan.length > 0) {
                planHtml = `
                    <div class="task-steps-box" style="margin-top: 8px; background: rgba(0,0,0,0.3); border: 1px solid var(--border-color); border-radius: 6px; padding: 10px;">
                        <div style="font-size: 0.75rem; font-weight: 700; color: var(--accent-cyan); margin-bottom: 6px; letter-spacing: 0.5px;">
                            WORKFLOW PLAN (${task.plan.length} STEPS)${task.replanned ? ' &bull; <span style="color: var(--status-warning);">[REPLANNED]</span>' : ''}:
                        </div>
                        ${task.plan.map(s => {
                            const icon = s.status === 'completed' ? '✔' : (s.status === 'failed' ? '✖' : (s.status === 'awaiting_confirmation' ? '⏳' : '▶'));
                            const color = s.status === 'completed' ? 'var(--status-success)' : (s.status === 'failed' ? 'var(--status-danger)' : (s.status === 'awaiting_confirmation' ? 'var(--status-warning)' : 'var(--text-muted)'));
                            return `
                                <div style="display: flex; align-items: center; justify-content: space-between; font-size: 0.8rem; padding: 4px 0; border-bottom: 1px solid rgba(255,255,255,0.04);">
                                    <div>
                                        <span style="color: ${color}; font-weight: 700;">${icon} Step ${s.step_id}:</span>
                                        <span style="color: var(--text-main); margin-left: 6px;">${escapeHtml(s.description)}</span>
                                    </div>
                                    <span class="badge" style="font-size: 0.7rem;">${s.tool_name || s.agent_role}</span>
                                </div>
                            `;
                        }).join('')}
                    </div>
                `;
            }

            chatMessages.innerHTML += `
                <div class="message-row assistant">
                    <div class="message-header" style="display: flex; flex-wrap: wrap; gap: 6px; align-items: center;">
                        <strong>A.O.R.I.S.</strong> &bull;
                        <span>Task ${task.task_id}</span>
                        ${orchBadge}
                        ${specialistBadge}
                        ${synthBadge}
                        ${modeBadge}
                        ${wfBadge}
                        ${cxBadge}
                    </div>
                    <div class="message-bubble" style="white-space: pre-wrap;">${escapeHtml(task.final_response || "Task processed.")}</div>
                    ${downloadHtml}
                    ${planHtml}
                    ${verificationHtml}
                </div>
            `;


            // Check if confirmation is required
            const confirmArea = document.getElementById('confirmation-area');
            if (task.state === 'awaiting_confirmation' && task.pending_confirmation) {
                confirmArea.innerHTML = `
                    <div class="confirmation-banner">
                        <div>
                            <strong style="color: var(--status-warning);">CONFIRMATION REQUIRED:</strong>
                            <div>${escapeHtml(task.pending_confirmation.prompt)}</div>
                        </div>
                        <div style="display: flex; gap: 8px;">
                            <button class="btn" onclick="approveTask('${task.task_id}')">Approve</button>
                            <button class="btn btn-danger" onclick="denyTask('${task.task_id}')">Deny</button>
                        </div>
                    </div>
                `;
            } else {
                confirmArea.innerHTML = '';
            }

            // Update side panel memories
            const memContainer = document.getElementById('retrieved-memories-container');
            if (data.memories_used && data.memories_used.length > 0) {
                memContainer.innerHTML = data.memories_used.map(m => `
                    <div class="memory-chip">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                            <span class="badge badge-cyan">${m.category}</span>
                            <span style="color: var(--status-success); font-weight: 600;">Conf: ${(m.confidence * 100).toFixed(0)}%</span>
                        </div>
                        <div>${escapeHtml(m.content)}</div>
                    </div>
                `).join('');
            } else {
                memContainer.innerHTML = `<div style="font-size: 0.8rem; color: var(--text-muted);">No memories required for this interaction.</div>`;
            }
        }

        async function approveTask(taskId) {
            await secureFetch(`/api/tasks/${taskId}/approve`, { method: 'POST' });
            document.getElementById('confirmation-area').innerHTML = '';
            sendMessage(); // Refresh chat
        }

        async function denyTask(taskId) {
            await secureFetch(`/api/tasks/${taskId}/deny`, { method: 'POST' });
            document.getElementById('confirmation-area').innerHTML = '';
        }

        async function loadMemories(query = '') {
            const url = query ? `/api/memories?query=${encodeURIComponent(query)}` : '/api/memories';
            const res = await secureFetch(url);
            const memories = await res.json();
            const container = document.getElementById('memory-cards-container');

            if (!Array.isArray(memories) || memories.length === 0) {
                container.innerHTML = '<div style="color: var(--text-muted);">No persistent memories stored yet.</div>';
                return;
            }

            container.innerHTML = memories.map(m => `
                <div class="memory-card">
                    <div class="card-header">
                        <span class="badge badge-cyan">${m.category}</span>
                        <span class="badge badge-success">${m.fact_status}</span>
                    </div>
                    <div class="card-body">${escapeHtml(m.content)}</div>
                    <div class="card-footer">
                        <span>Accesses: ${m.access_count}</span>
                        <div>
                            <button class="btn btn-secondary" style="padding: 4px 8px; font-size: 0.75rem;" onclick="deleteMemory('${m.id}')">Delete</button>
                            <button class="btn btn-danger" style="padding: 4px 8px; font-size: 0.75rem; margin-left: 4px;" onclick="purgeMemory('${m.id}')">Purge</button>
                        </div>
                    </div>
                </div>
            `).join('');
        }

        function searchMemories() {
            const query = document.getElementById('memory-search-input').value;
            loadMemories(query);
        }

        async function openAddMemoryModal() {
            const content = prompt("Enter factual memory statement or user preference:");
            if (!content) return;
            const category = prompt("Enter category (preference, project, rule, entity, general):", "general") || "general";
            await secureFetch('/api/memories', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ content: content, category: category, tags: [] })
            });
            loadMemories();
        }

        async function deleteMemory(id) {
            if (confirm("Soft-delete memory? It can still be retained for audit trails.")) {
                await secureFetch(`/api/memories/${id}?soft=true`, { method: 'DELETE' });
                loadMemories();
            }
        }

        async function purgeMemory(id) {
            if (confirm("Permanently purge this memory from disk? This cannot be undone.")) {
                await secureFetch(`/api/memories/${id}?soft=false`, { method: 'DELETE' });
                loadMemories();
            }
        }

        async function loadTasks() {
            const res = await secureFetch('/api/tasks');
            const tasks = await res.json();
            const tbody = document.getElementById('task-table-body');
            if (!Array.isArray(tasks)) return;
            tbody.innerHTML = tasks.map(t => `
                <tr>
                    <td><code>${t.task_id}</code></td>
                    <td>${escapeHtml(t.user_request)}</td>
                    <td><span class="badge ${t.state === 'completed' ? 'badge-success' : (t.state === 'failed' ? 'badge-danger' : 'badge-warning')}">${t.state}</span></td>
                    <td>${t.verification ? t.verification.verdict.toUpperCase() : 'N/A'}</td>
                    <td>${new Date(t.created_at).toLocaleTimeString()}</td>
                    <td>
                        ${t.state === 'awaiting_confirmation' ? `<button class="btn" style="padding: 4px 8px; font-size: 0.75rem;" onclick="approveTask('${t.task_id}')">Approve</button>` : ''}
                    </td>
                </tr>
            `).join('');
        }

        async function checkProviderHealth() {
            const res = await secureFetch('/api/provider/health');
            const data = await res.json();
            const badge = document.getElementById('provider-badge');
            badge.innerText = `Provider: ${data.provider_name} (${data.status})`;
            badge.className = data.is_live ? "badge badge-success" : "badge badge-warning";

            const alertContainer = document.getElementById('offline-alert-container');
            if (!data.is_live) {
                alertContainer.innerHTML = `
                    <div style="background: rgba(255, 171, 0, 0.12); border-bottom: 1px solid var(--status-warning); color: #ffca28; padding: 10px 24px; font-size: 0.85rem; display: flex; align-items: center; justify-content: space-between;">
                        <span>⚠️ <strong>OFFLINE FALLBACK ACTIVE:</strong> Operating in local deterministic mode. Configure <code>NEBIUS_API_KEY</code> in <code>.env</code> to activate live NVIDIA Nemotron inference.</span>
                        <span class="badge badge-warning">${escapeHtml(data.status)}</span>
                    </div>
                `;
            } else {
                alertContainer.innerHTML = `
                    <div style="background: rgba(0, 230, 118, 0.1); border-bottom: 1px solid var(--status-success); color: #69f0ae; padding: 10px 24px; font-size: 0.85rem;">
                        🌐 <strong>LIVE INFERENCE ACTIVE:</strong> Connected to ${escapeHtml(data.provider_name)} (${escapeHtml(data.model)}). Verified probe latency: ${data.latency_ms || 0}ms.
                    </div>
                `;
            }

            document.getElementById('provider-health-details').innerHTML = `
                <strong>Active Provider:</strong> ${escapeHtml(data.provider_name)}<br>
                <strong>Target Model:</strong> ${escapeHtml(data.model)}<br>
                <strong>Status Code:</strong> <span class="badge ${data.is_live ? 'badge-success' : 'badge-warning'}">${escapeHtml(data.status)}</span><br>
                <strong>Live Verified:</strong> ${data.is_live ? 'YES' : 'NO'}<br>
                ${data.latency_ms ? `<strong>Probe Latency:</strong> ${data.latency_ms} ms<br>` : ''}
                ${data.endpoint_url ? `<strong>Endpoint URL:</strong> ${escapeHtml(data.endpoint_url)}<br>` : ''}
                <strong>Diagnostic Note:</strong> ${escapeHtml(data.message)}
            `;
        }

        async function loadSettings() {
            checkProviderHealth();
            const res = await secureFetch('/api/settings');
            const data = await res.json();
            document.getElementById('active-config-details').innerText = JSON.stringify(data, null, 2);
        }

        async function checkMultiModelStatus() {
            try {
                const res = await secureFetch('/api/providers/status');
                if (!res.ok) return;
                const data = await res.json();
                const providers = data.providers || {};

                function updatePill(id, prov) {
                    const el = document.getElementById(id);
                    if (!el || !prov) return;
                    const st = prov.status;
                    let color = "var(--text-muted)";
                    let bg = "rgba(255,255,255,0.05)";
                    let border = "var(--border-color)";
                    if (st === 'available_verified') {
                        color = "var(--status-success)";
                        bg = "rgba(0, 230, 118, 0.12)";
                        border = "rgba(0, 230, 118, 0.4)";
                    } else if (st === 'rate_limited') {
                        color = "var(--status-warning)";
                        bg = "rgba(255, 171, 0, 0.12)";
                        border = "rgba(255, 171, 0, 0.4)";
                    } else if (st === 'authentication_failed') {
                        color = "var(--status-danger)";
                        bg = "rgba(255, 61, 113, 0.12)";
                        border = "rgba(255, 61, 113, 0.4)";
                    }
                    el.style.color = color;
                    el.style.background = bg;
                    el.style.borderColor = border;
                    el.title = `${prov.display_name} (${prov.model}): ${st.toUpperCase()}`;
                }

                updatePill('pill-nemotron', providers.nemotron);
                updatePill('pill-openai', providers.openai);
                updatePill('pill-anthropic', providers.anthropic);
                updatePill('pill-google', providers.google);
            } catch (err) {
                console.warn("Could not refresh multi-model status:", err);
            }
        }

        function escapeHtml(text) {
            if (!text) return '';
            return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
        }

        // Initialize on load
        window.onload = () => {
            updateAuthBadge();
            checkProviderHealth();
            checkMultiModelStatus();
            setInterval(checkMultiModelStatus, 20000);
        };
    </script>
</body>
</html>
"""
