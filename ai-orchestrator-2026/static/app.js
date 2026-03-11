/**
 * AI Orchestrator 2026 — Frontend Application
 * =============================================
 * SSE streaming client, DOM updates, tabs, score animations
 */

// ============================================================================
// STATE
// ============================================================================

let currentEventSource = null;
let isRunning = false;

// ============================================================================
// INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    loadSettings();
    loadKnowledgeBase();
    loadHistory();
    loadChromaStats();
    loadAgents(); // V2
    loadExternalCatalog();

    // Enter key to submit
    document.getElementById('promptInput').addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            startOrchestration();
        }
    });
});

// ============================================================================
// ORCHESTRATION (SSE)
// ============================================================================

function startOrchestration() {
    const input = document.getElementById('promptInput');
    const request = input.value.trim();
    if (!request || isRunning) return;

    isRunning = true;
    updateUI_running(true);
    resetPipeline();
    clearResults();
    addLog('system', `Запрос: "${request.substring(0, 80)}..."`);

    const maxIter = parseInt(document.getElementById('cfgIterations').value) || 3;
    const threshold = parseFloat(document.getElementById('cfgThreshold').value) || 0.85;

    document.getElementById('iterMax').textContent = maxIter;

    // POST + SSE
    fetch('/api/orchestrate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            request: request,
            max_iterations: maxIter,
            threshold: threshold,
            external_agent_url: document.getElementById('runMode').value === 'external'
                ? document.getElementById('externalUrl').value.trim()
                : null
        }),
    }).then(response => {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        function read() {
            reader.read().then(({ done, value }) => {
                if (done) {
                    finishOrchestration();
                    return;
                }

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop(); // keep incomplete line

                let eventType = '';
                let eventData = '';

                for (const line of lines) {
                    if (line.startsWith('event: ')) {
                        eventType = line.substring(7).trim();
                    } else if (line.startsWith('data: ')) {
                        eventData = line.substring(6).trim();
                        if (eventType && eventData) {
                            try {
                                const data = JSON.parse(eventData);
                                handleSSEEvent(eventType, data);
                            } catch (e) {
                                console.warn('SSE parse error:', e);
                            }
                            eventType = '';
                            eventData = '';
                        }
                    }
                }

                read();
            }).catch(err => {
                console.error('Stream error:', err);
                addLog('error', `Ошибка потока: ${err.message}`);
                finishOrchestration();
            });
        }

        read();
    }).catch(err => {
        console.error('Fetch error:', err);
        addLog('error', `Ошибка подключения: ${err.message}`);
        finishOrchestration();
    });
}

function stopOrchestration() {
    if (currentEventSource) {
        currentEventSource.close();
        currentEventSource = null;
    }
    finishOrchestration();
    addLog('system', 'Остановлено пользователем');
}

function finishOrchestration() {
    isRunning = false;
    updateUI_running(false);
    // Refresh sidebar data
    loadHistory();
    loadChromaStats();
}

// ============================================================================
// SSE EVENT HANDLERS
// ============================================================================

function handleSSEEvent(type, data) {
    switch (type) {
        case 'progress':
            handleProgress(data);
            break;
        case 'router':
            handleRouter(data);
            break;
        case 'generator':
            handleGenerator(data);
            break;
        case 'evaluator':
            handleEvaluator(data);
            break;
        case 'optimizer':
            handleOptimizer(data);
            break;
        case 'red_team':
            handleRedTeam(data);
            break;
        case 'complete':
            handleComplete(data);
            break;
        case 'error':
            handleError(data);
            break;
    }
}

function handleProgress(data) {
    const step = data.step;
    const msg = data.message;

    // Activate pipeline step
    activateStep(step);

    // Show progress message
    const progressEl = document.getElementById('progressMsg');
    progressEl.textContent = msg;
    progressEl.classList.add('visible');

    if (data.iteration) {
        document.getElementById('iterNum').textContent = data.iteration;
        document.getElementById('iterationBadge').style.display = '';
    }

    // Special handling for MoE agents in generator step
    if (step === 'generator' && data.iteration) {
        // Just updated the text in logs, handleGenerator will do final update
    }

    addLog(step, msg);
}

function handleRouter(data) {
    completeStep('router');
    const target = data.routing_target || data.request_type;
    document.getElementById('step-router-detail').textContent = target;
    addLog('router', `Тип: ${data.request_type}, Назначение: ${target}`);
    if (data.reasoning) {
        addLog('router', `Обоснование: ${data.reasoning}`);
    }
}

function handleGenerator(data) {
    completeStep('generator');
    const draftsText = data.drafts.map(d => `${d.agent} (${d.latency_ms}ms)`).join(', ');
    const targetLabel = data.target === 'external' ? 'Внешний агент' : `${data.drafts.length} черновиков`;
    document.getElementById('step-generator-detail').textContent = targetLabel;
    addLog('generator', `Итерация ${data.iteration}: ${draftsText}`);
}

function handleEvaluator(data) {
    completeStep('evaluator');
    const scoreText = (data.score * 100).toFixed(0) + '%';
    document.getElementById('step-evaluator-detail').textContent =
        `${scoreText} ${data.approved ? '✓' : '✗'}`;
    addLog('evaluator', `Score: ${data.score.toFixed(2)}, Approved: ${data.approved}`);

    // Update live score display
    updateScoreCard('scoreFulfill', 'barFulfill', data.task_fulfillment);
    updateScoreCard('scoreRAG', 'barRAG', data.rag_accuracy);
}

function handleOptimizer(data) {
    completeStep('optimizer');
    let msg = `Итерация ${data.iteration}: оптимизация завершена.`;
    if (data.agent_updated) {
        msg += " 🚀 Инструкции агента улучшены для самосовершенствования!";
        loadAgents(); // Refresh registry UI
    }
    document.getElementById('step-optimizer-detail').textContent = data.agent_updated ? 'Self-Improved' : 'Optimized';
    addLog('optimizer', msg);
}

function handleRedTeam(data) {
    completeStep('red_team');
    const robPct = (data.robustness * 100).toFixed(0);
    const robText = robPct + '%';
    document.getElementById('step-red_team-detail').textContent =
        `${data.vulnerabilities} уязв., ${robText}`;
    addLog('red_team', `Уязвимостей: ${data.vulnerabilities}, Robustness: ${robText}`);

    // Update robustness score card
    updateScoreCard('scoreRobust', 'barRobust', data.robustness);

    if (data.routing_target === 'optimizer') {
        addLog('red_team', `⚠️ Критические уязвимости! Возврат к оптимизации...`);
    }
}

function handleComplete(data) {
    document.getElementById('progressMsg').classList.remove('visible');

    // Show results
    document.getElementById('resultsEmpty').style.display = 'none';
    document.getElementById('resultsContent').style.display = 'block';

    // Final prompt
    document.getElementById('resultPrompt').textContent = data.final_prompt;

    // Scores
    updateScoreCard('scoreFinal', 'barFinal', data.score);
    updateScoreCard('scoreFulfill', 'barFulfill', data.score);

    // Score badge
    const scoreBadge = document.getElementById('resultScore');
    scoreBadge.style.display = '';
    document.getElementById('scoreVal').textContent = data.score.toFixed(2);

    // Robustness (from red_team data stored)
    const robustEl = document.getElementById('scoreRobust');
    if (robustEl.textContent === '—') {
        updateScoreCard('scoreRobust', 'barRobust', 1.0);
    }

    addLog('system', `✅ Завершено! Score: ${data.score.toFixed(2)}, Approved: ${data.approved}`);

    // Refresh sidebar
    setTimeout(() => {
        loadHistory();
        loadChromaStats();
    }, 500);
}

function handleError(data) {
    addLog('error', `❌ ${data.message}`);
    finishOrchestration();
}

// ============================================================================
// PIPELINE VISUALIZATION
// ============================================================================

const STEPS = ['router', 'generator', 'evaluator', 'optimizer', 'red_team'];

function resetPipeline() {
    STEPS.forEach(s => {
        const el = document.getElementById(`step-${s}`);
        el.classList.remove('active', 'completed', 'error');
        document.getElementById(`step-${s}-detail`).textContent = '';
    });
    document.getElementById('progressMsg').classList.remove('visible');
    document.getElementById('iterationBadge').style.display = 'none';
}

function activateStep(stepId) {
    STEPS.forEach(s => {
        const el = document.getElementById(`step-${s}`);
        if (s === stepId) {
            el.classList.add('active');
            el.classList.remove('completed');
        }
    });
}

function completeStep(stepId) {
    const el = document.getElementById(`step-${stepId}`);
    el.classList.remove('active');
    el.classList.add('completed');
}

// ============================================================================
// SCORE CARDS
// ============================================================================

function updateScoreCard(valueId, barId, value) {
    const el = document.getElementById(valueId);
    const bar = document.getElementById(barId);
    const pct = (value * 100);

    el.textContent = pct.toFixed(0) + '%';
    el.className = 'score-value ' + (pct >= 80 ? 'good' : pct >= 50 ? 'warning' : 'danger');
    bar.style.width = pct + '%';
    bar.style.background = pct >= 80
        ? 'linear-gradient(90deg, #22c55e, #16a34a)'
        : pct >= 50
            ? 'linear-gradient(90deg, #f59e0b, #d97706)'
            : 'linear-gradient(90deg, #ef4444, #dc2626)';
}

// ============================================================================
// RESULTS
// ============================================================================

function clearResults() {
    document.getElementById('resultsEmpty').style.display = '';
    document.getElementById('resultsContent').style.display = 'none';
    document.getElementById('resultScore').style.display = 'none';
    document.getElementById('resultPrompt').textContent = '';

    // Reset scores
    ['scoreFinal', 'scoreFulfill', 'scoreRAG', 'scoreRobust'].forEach(id => {
        document.getElementById(id).textContent = '—';
        document.getElementById(id).className = 'score-value';
    });
    ['barFinal', 'barFulfill', 'barRAG', 'barRobust'].forEach(id => {
        document.getElementById(id).style.width = '0';
    });
}

function copyPrompt() {
    const text = document.getElementById('resultPrompt').textContent;
    navigator.clipboard.writeText(text).then(() => {
        const btn = document.querySelector('.copy-btn');
        btn.textContent = '✓ Скопировано!';
        setTimeout(() => { btn.textContent = '📋 Копировать'; }, 2000);
    });
}

// ============================================================================
// UI STATE
// ============================================================================

function updateUI_running(running) {
    document.getElementById('runBtn').disabled = running;
    document.getElementById('runBtn').innerHTML = running
        ? '<span class="spinner"></span> Работает...'
        : '🚀 Запуск';
    document.getElementById('stopBtn').style.display = running ? '' : 'none';

    // Disable inputs while running
    document.getElementById('promptInput').disabled = running;
    document.getElementById('runMode').disabled = running;
}

function toggleExternalUrl() {
    const mode = document.getElementById('runMode').value;
    document.getElementById('externalUrlWrap').style.display = mode === 'external' ? 'block' : 'none';
}

// ============================================================================
// TABS
// ============================================================================

function switchTab(tabName) {
    // Deactivate all
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

    // Activate selected
    document.querySelector(`.tab-btn[onclick*="${tabName}"]`).classList.add('active');
    document.getElementById(`tab-${tabName}`).classList.add('active');
}

// ============================================================================
// KNOWLEDGE BASE
// ============================================================================

async function loadKnowledgeBase() {
    try {
        const resp = await fetch('/api/kb');
        const data = await resp.json();

        const container = document.getElementById('kbList');
        if (!data.nodes || data.nodes.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <div class="empty-icon">📚</div>
                    <div class="empty-text">База знаний пуста</div>
                </div>`;
            return;
        }

        const icons = { music_theory: '🎵', psychology: '🧠', prompt_engineering: '🎯', creative_writing: '✍️', technical_docs: '📐' };

        container.innerHTML = data.nodes.map(node => `
            <div class="kb-node slide-in" onclick="queryKB('${node}')">
                <div class="kb-node-title">
                    ${icons[node] || '📄'} ${formatNodeName(node)}
                </div>
                <div class="kb-node-preview">${data.data[node] || 'Нет данных'}</div>
            </div>
        `).join('');
    } catch (e) {
        console.warn('KB load error:', e);
    }
}

function formatNodeName(name) {
    return name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

async function queryKB(nodeId) {
    try {
        const resp = await fetch('/api/kb/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: nodeId }),
        });
        const data = await resp.json();
        addLog('system', `📚 KB[${nodeId}]: ${data.result.substring(0, 100)}...`);
    } catch (e) {
        console.warn('KB query error:', e);
    }
}

// ============================================================================
// HISTORY
// ============================================================================

async function loadExternalCatalog() {
    try {
        const resp = await fetch('/api/external/catalog');
        const catalog = await resp.json();
        const select = document.getElementById('external-agent-presets');
        if (!select) return;

        select.innerHTML = '<option value="">-- Выберите из каталога --</option>';
        for (const [name, url] of Object.entries(catalog)) {
            const opt = document.createElement('option');
            opt.value = url;
            opt.textContent = name;
            select.appendChild(opt);
        }
    } catch (e) {
        console.warn('External catalog load error:', e);
    }
}
async function loadHistory() {
    try {
        const resp = await fetch('/api/history');
        const sessions = await resp.json();

        state.history = sessions;
        updateHistoryList();

    } catch (e) {
        console.warn('History load error:', e);
    }
}

function updateHistoryList() {
    const container = document.getElementById('historyList');
    if (!container) return;

    if (!state.history || state.history.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">📜</div>
                <div class="empty-text">История сессий пуста</div>
            </div>`;
        return;
    }

    container.innerHTML = state.history.map(s => `
        <div class="history-item slide-in" onclick="loadSessionDetail('${s.session_id}')">
            <div class="history-dot ${s.approved ? 'approved' : 'pending'}"></div>
            <div class="history-meta">
                <div class="history-score">${(s.best_score || 0).toFixed(2)} ${s.approved ? '✓' : '✗'}</div>
                <div class="history-request">${s.request || '—'}</div>
                <div class="history-time">${formatTime(s.completed_at)} · ${s.versions} версий</div>
            </div>
        </div>
    `).join('');
}

async function loadSessionDetail(sessionId) {
    try {
        const response = await fetch(`/api/history/${sessionId}`);
        const versions = await response.json();

        if (versions.length > 0) {
            // Берем последнюю одобренную или просто последнюю
            const latest = versions[versions.length - 1];
            const rollbackResp = await fetch(`/api/rollback/${latest.version_id}`, { method: 'POST' });
            const rollbackData = await rollbackResp.json();

            if (rollbackData.success) {
                const fullState = rollbackData.state;

                // Восстанавливаем UI
                document.getElementById('promptInput').value = fullState.request; // Assuming promptInput is the user request field
                document.getElementById('resultPrompt').textContent = fullState.final_prompt || ""; // Assuming resultPrompt is the final prompt field

                // Переключаемся на вкладку результатов или главную
                switchTab('orchestration'); // Or a more appropriate tab
                alert(`Сессия ${sessionId.substring(0, 8)} загружена. Вы можете продолжить работу.`);
            } else {
                console.error("Rollback failed:", rollbackData.message);
                alert("Не удалось загрузить сессию.");
            }
        } else {
            alert("Нет данных для этой сессии.");
        }
    } catch (e) {
        console.error("Error loading session:", e);
        alert("Ошибка при загрузке сессии.");
    }
}

function formatTime(isoString) {
    if (!isoString) return '—';
    try {
        const d = new Date(isoString);
        return d.toLocaleString('ru-RU', { hour: '2-digit', minute: '2-digit', day: '2-digit', month: '2-digit' });
    } catch (e) {
        return isoString;
    }
}

// ============================================================================
// AGENT REGISTRY (V2)
// ============================================================================

async function loadAgents() {
    try {
        const resp = await fetch('/api/agents');
        const agents = await resp.json();

        state.agents = agents;
        updateAgentList();

    } catch (e) {
        console.warn('Agents load error:', e);
    }
}

function updateAgentList() {
    const container = document.getElementById('agentGrid');
    if (!container) return;

    if (!state.agents || state.agents.length === 0) {
        container.innerHTML = `<div class="empty-state">Реестр пуст</div>`;
        return;
    }

    const icons = { creative_expert: '🎨', struct_expert: '📐', tone_expert: '🎭', adversary_agent: '🔴' };

    container.innerHTML = state.agents.map(a => `
        <div class="agent-card slide-in">
            <div class="agent-icon" style="background:var(--bg-secondary)">${icons[a.name] || '🤖'}</div>
            <div class="agent-info">
                <div class="agent-name">${t(a.name)}</div>
                <div class="agent-desc">${a.role || 'Специализированный агент'}</div>
                <div class="agent-stats">
                    <span class="agent-stat-pill version">v${a.version || 1}</span>
                    <span class="agent-stat-pill score">${((a.avg_score || 0) * 100).toFixed(0)}% avg</span>
                </div>
            </div>
        </div>
    `).join('');
}

function showAddAgentModal() {
    document.getElementById('addAgentModal').classList.add('active');
}

function hideAddAgentModal() {
    document.getElementById('addAgentModal').classList.remove('active');
}

async function addAgent() {
    const name = document.getElementById('newAgentName').value.trim();
    const role = document.getElementById('newAgentRole').value.trim();
    const instruction = document.getElementById('agent-instructions').value.trim();
    const kbFile = document.getElementById('agent-kb-file').files[0];

    if (!name || !instruction) {
        alert("Имя и инструкции обязательны!");
        return;
    }

    let kbContent = "";
    if (kbFile) {
        kbContent = await kbFile.text();
    }

    try {
        const resp = await fetch('/api/agents/add', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                name,
                role,
                instruction,
                kb_content: kbContent,
                version: 1,
                task_types: []
            })
        });
        const res = await resp.json();
        if (res.success) {
            addLog('system', `Агент ${name} добавлен в реестр`);
            hideAddAgentModal();
            loadAgents();
            // Clear form
            ['newAgentName', 'newAgentRole', 'agent-instructions', 'agent-kb-file'].forEach(id => {
                const element = document.getElementById(id);
                if (element) {
                    if (element.type === 'file') {
                        element.value = ''; // Clear file input
                    } else {
                        element.value = '';
                    }
                }
            });
        } else {
            addLog('error', `Ошибка добавления агента: ${res.message || 'Неизвестная ошибка'}`);
        }
    } catch (e) {
        addLog('error', `Ошибка добавления агента: ${e.message}`);
    }
}

// ============================================================================
// CHROMADB STATS
// ============================================================================

async function loadChromaStats() {
    try {
        const resp = await fetch('/api/chromadb/stats');
        const stats = await resp.json();

        document.getElementById('statKB').textContent = stats.knowledge_base?.count ?? '—';
        document.getElementById('statPrompts').textContent = stats.prompt_versions?.count ?? '—';
        document.getElementById('statMode').textContent = stats.mode === 'chromadb' ? 'Chroma' : 'File';

        // Count sessions
        const histResp = await fetch('/api/history');
        const sessions = await histResp.json();
        document.getElementById('statSessions').textContent = Array.isArray(sessions) ? sessions.length : '—';
    } catch (e) {
        console.warn('Stats load error:', e);
    }
}

// ============================================================================
// SETTINGS
// ============================================================================

async function loadSettings() {
    try {
        const resp = await fetch('/api/settings');
        const settings = await resp.json();

        document.getElementById('cfgIterations').value = settings.max_iterations || 3;
        document.getElementById('cfgThreshold').value = settings.approval_threshold || 0.85;

        // Mode indicator
        const modeText = document.getElementById('modeText');
        modeText.textContent = settings.llm_mode || 'REAL';
    } catch (e) {
        console.warn('Settings load error:', e);
    }
}

// ============================================================================
// LOGGING
// ============================================================================

function addLog(tag, message) {
    const container = document.getElementById('logEntries');
    const now = new Date().toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit', second: '2-digit' });

    const entry = document.createElement('div');
    entry.className = 'log-entry fade-in';
    entry.innerHTML = `
        <span class="log-time">[${now}]</span>
        <span class="log-tag ${tag}">[${tag}]</span>
        <span>${escapeHtml(message)}</span>
    `;

    container.appendChild(entry);
    container.scrollTop = container.scrollHeight;
}

function clearLogs() {
    document.getElementById('logEntries').innerHTML = `
        <div class="log-entry">
            <span class="log-time">[init]</span>
            <span>Лог очищен</span>
        </div>`;
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}
