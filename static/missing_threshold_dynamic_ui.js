/**
 * DYNAMIC MISSING THRESHOLD SYSTEM — FIXED VERSION
 *
 * Fixes:
 * 1. DOM-level lock (data-locked="true") prevents backend polling from
 *    overriding auto-absent status even when camera re-detects the student.
 * 2. Missing timer STOPS once threshold is hit — no more counting after absent.
 * 3. Right-click manual override always removes the lock (manual wins).
 * 4. is_locked sent to DB via manage_student API so backend also honours it.
 */

// ============================================
// GLOBAL STATE
// ============================================
window.missingThresholdConfig = {
    enabled: true,
    thresholdMinutes: 1,
    checkIntervalSeconds: 15,
    lockAbsentStatus: true,
    notifyOnChange: true
};

let missingStudentTimestamps = new Map(); // studentId -> { timestamp, locked }
let lockedAbsentStudents = new Set();     // studentIds locked as absent
let missingCheckIntervalId = null;


// ============================================
// SETTINGS HTML (injected into modal)
// ============================================
const missingThresholdSettingsHTML = `
<div class="settings-section" style="border-top: 2px solid #e0e0e0; margin-top: 20px; padding-top: 20px;">
    <div class="section-header" style="display: flex; align-items: center; gap: 10px; margin-bottom: 15px;">
        <span class="material-icons" style="color: #ff9800; font-size: 28px;">person_off</span>
        <h4 style="margin: 0; color: #333;">Missing Threshold Settings</h4>
    </div>

    <div class="form-group" style="background: #fff3e0; padding: 15px; border-radius: 8px; border-left: 4px solid #ff9800; margin-bottom: 20px;">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div style="flex: 1;">
                <label style="font-weight: 600; margin: 0; display: flex; align-items: center; gap: 8px;">
                    <span class="material-icons" style="color: #ff9800;">notifications_active</span>
                    Enable Missing Threshold Auto-Absent
                </label>
                <div style="font-size: 12px; color: #666; margin-top: 4px;">
                    Automatically mark students as ABSENT when they exceed missing threshold
                </div>
            </div>
            <label class="switch">
                <input type="checkbox" id="missingThresholdEnabled" checked onchange="toggleMissingThreshold(this.checked)">
                <span class="slider round"></span>
            </label>
        </div>
    </div>

    <div class="form-group" style="margin-bottom: 20px;">
        <label style="font-weight: 600; margin-bottom: 10px; display: block;">
            <span class="material-icons" style="vertical-align: middle; font-size: 18px;">timer</span>
            Missing Threshold Duration
        </label>
        <div style="background: #f5f5f5; padding: 20px; border-radius: 8px;">
            <div style="text-align: center; margin-bottom: 15px;">
                <div style="font-size: 14px; color: #666; margin-bottom: 5px;">Current Threshold</div>
                <div style="font-size: 32px; font-weight: bold; color: #ff9800;">
                    <span id="currentMissingThreshold">1</span> minute<span id="missingThresholdPlural">s</span>
                </div>
                <div style="font-size: 12px; color: #999; margin-top: 5px;">
                    Students missing longer than this are automatically marked ABSENT (locked)
                </div>
            </div>
            <div style="margin-bottom: 20px;">
                <label style="font-size: 13px; color: #666; margin-bottom: 8px; display: block;">Adjust Threshold (1-30 minutes):</label>
                <input type="range" id="missingThresholdSlider" min="1" max="30" value="1" step="1"
                       style="width: 100%;" oninput="updateMissingThresholdDisplay(this.value)">
                <div style="display: flex; justify-content: space-between; font-size: 11px; color: #999; margin-top: 5px;">
                    <span>1 min</span><span>15 min</span><span>30 min</span>
                </div>
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px;">
                <button class="btn-quick-set" onclick="setMissingThreshold(1)">1 min</button>
                <button class="btn-quick-set" onclick="setMissingThreshold(3)">3 min</button>
                <button class="btn-quick-set" onclick="setMissingThreshold(5)">5 min</button>
                <button class="btn-quick-set" onclick="setMissingThreshold(10)">10 min</button>
            </div>
        </div>
    </div>

    <div class="form-group" style="margin-bottom: 20px;">
        <label style="font-weight: 600; margin-bottom: 10px; display: block;">
            <span class="material-icons" style="vertical-align: middle; font-size: 18px;">update</span>
            Check Frequency
        </label>
        <select id="missingCheckFrequency" class="form-select" onchange="updateCheckFrequency(this.value)">
            <option value="10">Every 10 seconds (High responsiveness)</option>
            <option value="15" selected>Every 15 seconds (Recommended)</option>
            <option value="30">Every 30 seconds (Low CPU usage)</option>
            <option value="60">Every 60 seconds (Minimal)</option>
        </select>
        <div style="font-size: 12px; color: #666; margin-top: 5px;">
            How often the system checks for missing threshold violations
        </div>
    </div>

    <div class="form-group" style="background: #ffebee; padding: 15px; border-radius: 8px; border-left: 4px solid #f44336; margin-bottom: 20px;">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div style="flex: 1;">
                <label style="font-weight: 600; margin: 0; display: flex; align-items: center; gap: 8px;">
                    <span class="material-icons" style="color: #f44336;">lock</span>
                    Lock ABSENT Status (Recommended)
                </label>
                <div style="font-size: 12px; color: #666; margin-top: 4px;">
                    Once auto-marked ABSENT, student stays absent even if camera detects them again.
                    Right-click the student to manually override.
                </div>
            </div>
            <label class="switch">
                <input type="checkbox" id="lockAbsentStatus" checked onchange="toggleLockAbsentStatus(this.checked)">
                <span class="slider round"></span>
            </label>
        </div>
    </div>

    <div class="form-group">
        <label style="font-weight: 600; margin-bottom: 10px; display: block;">
            <span class="material-icons" style="vertical-align: middle; font-size: 18px;">visibility</span>
            Currently Tracked Missing Students
        </label>
        <div id="trackedMissingStudents" style="background: #f5f5f5; padding: 15px; border-radius: 8px; min-height: 60px; max-height: 200px; overflow-y: auto;">
            <div style="text-align: center; color: #999; font-style: italic;">
                No students currently marked as missing
            </div>
        </div>
    </div>

    <div class="form-group" style="margin-top: 20px;">
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;">
            <div style="background: #e3f2fd; padding: 12px; border-radius: 8px; text-align: center;">
                <div style="font-size: 11px; color: #666; margin-bottom: 4px;">Missing</div>
                <div style="font-size: 24px; font-weight: bold; color: #2196f3;"><span id="statMissing">0</span></div>
            </div>
            <div style="background: #fff3e0; padding: 12px; border-radius: 8px; text-align: center;">
                <div style="font-size: 11px; color: #666; margin-bottom: 4px;">Tracked</div>
                <div style="font-size: 24px; font-weight: bold; color: #ff9800;"><span id="statTracked">0</span></div>
            </div>
            <div style="background: #ffebee; padding: 12px; border-radius: 8px; text-align: center;">
                <div style="font-size: 11px; color: #666; margin-bottom: 4px;">Auto-Absent</div>
                <div style="font-size: 24px; font-weight: bold; color: #f44336;"><span id="statAutoAbsent">0</span></div>
            </div>
        </div>
    </div>
</div>
`;

// ============================================
// CSS STYLES
// ============================================
const missingThresholdStyles = `
<style>
.switch { position: relative; display: inline-block; width: 50px; height: 24px; }
.switch input { opacity: 0; width: 0; height: 0; }
.slider { position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background-color: #ccc; transition: .4s; }
.slider:before { position: absolute; content: ""; height: 16px; width: 16px; left: 4px; bottom: 4px; background-color: white; transition: .4s; }
input:checked + .slider { background-color: #4CAF50; }
input:checked + .slider:before { transform: translateX(26px); }
.slider.round { border-radius: 24px; }
.slider.round:before { border-radius: 50%; }

.btn-quick-set { padding: 8px 12px; background: white; border: 2px solid #ff9800; color: #ff9800; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 13px; transition: all 0.3s; }
.btn-quick-set:hover { background: #ff9800; color: white; transform: translateY(-2px); box-shadow: 0 4px 8px rgba(255,152,0,0.3); }
.btn-quick-set:active { transform: translateY(0); }

input[type="range"] { -webkit-appearance: none; appearance: none; height: 8px; background: #ddd; border-radius: 5px; outline: none; }
input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 20px; height: 20px; background: #ff9800; cursor: pointer; border-radius: 50%; box-shadow: 0 2px 4px rgba(0,0,0,0.2); }
input[type="range"]::-moz-range-thumb { width: 20px; height: 20px; background: #ff9800; cursor: pointer; border-radius: 50%; box-shadow: 0 2px 4px rgba(0,0,0,0.2); border: none; }

.form-select { width: 100%; padding: 10px 12px; border: 1px solid #ddd; border-radius: 6px; font-size: 14px; background: white; cursor: pointer; }
.form-select:focus { outline: none; border-color: #ff9800; box-shadow: 0 0 0 3px rgba(255,152,0,0.1); }

.missing-student-card { display: flex; align-items: center; justify-content: space-between; padding: 10px; background: white; border-left: 4px solid #ff9800; border-radius: 6px; margin-bottom: 8px; }
.missing-student-card.warning { border-left-color: #ff5722; background: #fff3e0; }
.missing-student-card.locked-absent { border-left-color: #f44336; background: #ffebee; opacity: 0.85; }
</style>
`;

document.head.insertAdjacentHTML('beforeend', missingThresholdStyles);


// ============================================
// UI CONTROL FUNCTIONS
// ============================================

function updateMissingThresholdDisplay(minutes) {
    const display = document.getElementById('currentMissingThreshold');
    const plural  = document.getElementById('missingThresholdPlural');
    if (display) display.textContent = minutes;
    if (plural)  plural.textContent  = minutes == 1 ? '' : 's';
    window.missingThresholdConfig.thresholdMinutes = parseInt(minutes);
}

function setMissingThreshold(minutes) {
    const slider = document.getElementById('missingThresholdSlider');
    if (slider) slider.value = minutes;
    updateMissingThresholdDisplay(minutes);
    showModalNotification(`Missing threshold set to ${minutes} minute${minutes == 1 ? '' : 's'}`, 'success', 2000);
}

async function applyMissingThresholdSettings() {
    const threshold  = window.missingThresholdConfig.thresholdMinutes;
    const sessionId  = window._sessionData?.session_id;
    if (!sessionId) { showModalNotification('Session ID not found', 'error', 3000); return false; }

    try {
        const response = await fetch('/api/update_missing_threshold', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                missing_threshold_minutes: threshold,
                enabled: window.missingThresholdConfig.enabled,
                lock_absent: window.missingThresholdConfig.lockAbsentStatus
            })
        });
        const data = await response.json();
        if (data.success) {
            showModalNotification(`Missing threshold set to ${threshold} minute${threshold == 1 ? '' : 's'}`, 'success', 3000);
            restartMissingMonitoring();
            return true;
        } else {
            showModalNotification(data.message || 'Failed to update threshold', 'error', 3000);
            return false;
        }
    } catch (error) {
        showModalNotification('Network error updating threshold', 'error', 3000);
        return false;
    }
}

function toggleMissingThreshold(enabled) {
    window.missingThresholdConfig.enabled = enabled;
    if (enabled) { startMissingMonitoring(); showModalNotification('Missing threshold monitoring enabled', 'success', 2000); }
    else         { stopMissingMonitoring();  showModalNotification('Missing threshold monitoring disabled', 'warning', 2000); }
}

function toggleLockAbsentStatus(enabled) {
    window.missingThresholdConfig.lockAbsentStatus = enabled;
    showModalNotification(enabled ? 'Lock absent status enabled' : 'Lock absent status disabled', enabled ? 'success' : 'warning', 2000);
}

function updateCheckFrequency(seconds) {
    window.missingThresholdConfig.checkIntervalSeconds = parseInt(seconds);
    if (window.missingThresholdConfig.enabled) restartMissingMonitoring();
    showModalNotification(`Check frequency set to every ${seconds} seconds`, 'success', 2000);
}


// ============================================
// CORE MONITORING
// ============================================

function startMissingMonitoring() {
    if (missingCheckIntervalId) clearInterval(missingCheckIntervalId);
    const intervalMs = window.missingThresholdConfig.checkIntervalSeconds * 1000;
    missingCheckIntervalId = setInterval(() => {
        if (isSessionEnding || isDetectionStopped) return;
        checkMissingThresholdViolations();
    }, intervalMs);
    console.log(`🚀 Missing monitoring started (every ${window.missingThresholdConfig.checkIntervalSeconds}s)`);
}

function stopMissingMonitoring() {
    if (missingCheckIntervalId) { clearInterval(missingCheckIntervalId); missingCheckIntervalId = null; }
    console.log('⏹️ Missing monitoring stopped');
}

function restartMissingMonitoring() {
    stopMissingMonitoring();
    if (window.missingThresholdConfig.enabled) startMissingMonitoring();
}

/**
 * Main periodic check — runs every N seconds
 */
async function checkMissingThresholdViolations() {
    if (!window.missingThresholdConfig.enabled) return;

    const thresholdMs = window.missingThresholdConfig.thresholdMinutes * 60 * 1000;
    const now = new Date();
    const studentRows = document.querySelectorAll('.student-row');
    let violationsFound = 0;

    for (const row of studentRows) {
        const statusElement  = row.querySelector('.status');
        const studentId      = row.getAttribute('data-student-id');
        const studentName    = row.querySelector('.student-name')?.textContent?.replace(/\s*lock\s*/i, '').trim();

        if (!statusElement || !studentId) continue;

        const currentStatus = statusElement.textContent.toLowerCase().trim();

        // ── Already DOM-locked → enforce absent, skip timer logic ──
        if (row.getAttribute('data-locked') === 'true' || lockedAbsentStudents.has(studentId)) {
            if (currentStatus !== 'absent') {
                statusElement.className = 'status absent';
                statusElement.textContent = 'Absent';
            }
            // Clean up timer tracking if still present
            if (missingStudentTimestamps.has(studentId)) {
                missingStudentTimestamps.delete(studentId);
                updateTrackedMissingDisplay();
            }
            continue;
        }

        // ── Student is MISSING → track / check threshold ──
        if (currentStatus === 'missing') {
            if (!missingStudentTimestamps.has(studentId)) {
                missingStudentTimestamps.set(studentId, { timestamp: now });
                console.log(`📝 Tracking missing: ${studentName}`);
                updateTrackedMissingDisplay();
            }

            const trackData   = missingStudentTimestamps.get(studentId);
            const elapsedMs   = now - trackData.timestamp;
            const elapsedMins = Math.floor(elapsedMs / 60000);

            if (elapsedMs > thresholdMs) {
                console.log(`⚠️ THRESHOLD EXCEEDED: ${studentName} (${elapsedMins}m)`);
                await changeStudentToAbsentFromMissing(
                    studentId,
                    studentName,
                    `Missing for ${elapsedMins} minute${elapsedMins !== 1 ? 's' : ''}`
                );
                violationsFound++;
            }
        }
        // ── No longer missing → remove from tracking ──
        else if (missingStudentTimestamps.has(studentId)) {
            missingStudentTimestamps.delete(studentId);
            updateTrackedMissingDisplay();
        }
    }

    if (violationsFound > 0) {
        if (typeof updateStudentCount === 'function') updateStudentCount();
        updateStatistics();
    }

    updateTrackedMissingDisplay();
}

/**
 * Auto-change MISSING → ABSENT and lock the student in DOM + DB
 */
async function changeStudentToAbsentFromMissing(studentId, studentName, reason) {
    console.log(`🤖 AUTO-ABSENT: ${studentName} (${studentId}) — ${reason}`);

    // ── Immediately stop the timer by removing from map ──
    missingStudentTimestamps.delete(studentId);

    try {
        const res = await fetch('/api/manage_student', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                action: 'mark_present',
                student_data: {
                    student_id: studentId,
                    status: 'absent',
                    remarks: `Auto-absent: ${reason}`,
                    automatic: true,
                    from_missing: true,
                    // Tell DB to lock this record (is_locked column)
                    is_locked: window.missingThresholdConfig.lockAbsentStatus ? 1 : 0
                }
            })
        });

        const data = await res.json();

        if (data.success) {
            // ── Update the DOM row immediately ──
            const studentRow = document.querySelector(`.student-row[data-student-id="${studentId}"]`);
            if (studentRow) {
                const statusEl = studentRow.querySelector('.status');
                if (statusEl) {
                    statusEl.className = 'status absent';
                    statusEl.textContent = 'Absent';
                }

                if (window.missingThresholdConfig.lockAbsentStatus) {
                    // DOM-level lock — survives every updateStudentStatuses call
                    studentRow.setAttribute('data-locked', 'true');
                    studentRow.setAttribute('data-lock-reason', `Auto-absent: ${reason}`);

                    // Add lock icon to name if not already there
                    const nameEl = studentRow.querySelector('.student-name');
                    if (nameEl && !nameEl.querySelector('.lock-icon')) {
                        const icon = document.createElement('span');
                        icon.className = 'material-icons lock-icon';
                        icon.title = 'Auto-locked absent — right-click to override';
                        icon.style.cssText = 'font-size:14px;color:#f44336;vertical-align:middle;margin-left:4px;';
                        icon.textContent = 'lock';
                        nameEl.appendChild(icon);
                    }
                }
            }

            // In-memory lock (backup guard inside updateStudentStatuses)
            if (window.missingThresholdConfig.lockAbsentStatus) {
                lockedAbsentStudents.add(studentId);
            }

            showMissingThresholdNotification(studentId, studentName, reason);
            updateStatistics();
            updateTrackedMissingDisplay(); // refresh panel (card should now be gone)
        } else {
            console.error(`❌ Failed to mark absent: ${data.message}`);
        }
    } catch (err) {
        console.error('Error in changeStudentToAbsentFromMissing:', err);
    }
}

/**
 * Force a locked student back to absent (safety net for edge cases)
 */
async function forceStudentAbsent(studentId, studentName, reason) {
    try {
        await fetch('/api/manage_student', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                action: 'mark_present',
                student_data: { student_id: studentId, status: 'absent', remarks: reason, automatic: true, is_locked: 1 }
            })
        });
        const row = document.querySelector(`.student-row[data-student-id="${studentId}"]`);
        if (row) {
            const s = row.querySelector('.status');
            if (s) { s.className = 'status absent'; s.textContent = 'Absent'; }
            row.setAttribute('data-locked', 'true');
        }
    } catch (err) {
        console.error('Error forcing absent:', err);
    }
}


// ============================================
// UI UPDATE FUNCTIONS
// ============================================

/**
 * Show toast notification when student is auto-marked absent
 */
function showMissingThresholdNotification(studentId, studentName, reason) {
    const notification = document.createElement('div');
    notification.className = 'threshold-notification missing-threshold-notification';
    notification.innerHTML = `
        <div style="display:flex;align-items:center;gap:12px;padding:15px 20px;background:#ffebee;border-left:4px solid #f44336;box-shadow:0 4px 12px rgba(0,0,0,0.15);border-radius:8px;margin:10px;animation:slideIn 0.3s ease-out;">
            <span class="material-icons" style="color:#f44336;font-size:28px;">person_remove</span>
            <div style="flex:1;">
                <div style="font-weight:600;font-size:15px;color:#333;">
                    <strong>${studentName || studentId}</strong> marked as <strong style="color:#f44336;">ABSENT</strong>
                </div>
                <div style="font-size:13px;color:#666;margin-top:3px;">${reason}</div>
                ${window.missingThresholdConfig.lockAbsentStatus
                    ? '<div style="font-size:12px;color:#f44336;margin-top:3px;display:flex;align-items:center;gap:4px;"><span class="material-icons" style="font-size:14px;">lock</span> Locked — right-click student to override</div>'
                    : ''}
            </div>
            <button onclick="this.parentElement.parentElement.remove()" style="background:none;border:none;cursor:pointer;padding:4px;">
                <span class="material-icons" style="color:#999;">close</span>
            </button>
        </div>
    `;

    const studentList = document.getElementById('studentList');
    if (studentList) {
        studentList.insertBefore(notification, studentList.firstChild);
        setTimeout(() => {
            if (notification.parentElement) {
                notification.style.animation = 'slideOut 0.3s ease-in';
                setTimeout(() => notification.remove(), 300);
            }
        }, 12000);
    }
}

/**
 * Update the tracked-missing panel in settings.
 * Once a student is locked-absent their card shows "ABSENT" and the timer is stopped.
 */
function updateTrackedMissingDisplay() {
    const container = document.getElementById('trackedMissingStudents');
    if (!container) return;

    if (missingStudentTimestamps.size === 0) {
        container.innerHTML = `<div style="text-align:center;color:#999;font-style:italic;padding:20px;">No students currently marked as missing</div>`;
        return;
    }

    const now         = new Date();
    const thresholdMs = window.missingThresholdConfig.thresholdMinutes * 60 * 1000;
    let html          = '';

    missingStudentTimestamps.forEach((trackData, studentId) => {
        const row         = document.querySelector(`.student-row[data-student-id="${studentId}"]`);
        const studentName = row?.querySelector('.student-name')?.firstChild?.textContent?.trim() || studentId;
        const isLocked    = row?.getAttribute('data-locked') === 'true' || lockedAbsentStudents.has(studentId);

        if (isLocked) {
            // Timer stopped — show final absent badge
            html += `
                <div class="missing-student-card locked-absent">
                    <div style="flex:1;">
                        <div style="font-weight:600;margin-bottom:4px;display:flex;align-items:center;gap:6px;">
                            <span class="material-icons" style="font-size:16px;color:#f44336;">lock</span>
                            ${studentName}
                        </div>
                        <div style="font-size:12px;color:#f44336;font-weight:600;">Auto-marked ABSENT (locked)</div>
                        <div style="font-size:11px;color:#999;margin-top:2px;">Right-click student to manually override</div>
                    </div>
                    <span style="background:#f44336;color:white;font-size:11px;font-weight:700;padding:3px 8px;border-radius:12px;">ABSENT</span>
                </div>`;
            return;
        }

        const elapsedMs      = now - trackData.timestamp;
        const elapsedMinutes = Math.floor(elapsedMs / 60000);
        const elapsedSeconds = Math.floor((elapsedMs % 60000) / 1000);
        const percentComplete = Math.min((elapsedMs / thresholdMs) * 100, 100);
        const isWarning       = percentComplete > 75;
        const remaining       = Math.max(0, window.missingThresholdConfig.thresholdMinutes - elapsedMinutes);

        html += `
            <div class="missing-student-card ${isWarning ? 'warning' : ''}">
                <div style="flex:1;">
                    <div style="font-weight:600;margin-bottom:4px;">${studentName}</div>
                    <div style="font-size:12px;color:#666;">Missing for: ${elapsedMinutes}m ${elapsedSeconds}s</div>
                    <div style="margin-top:6px;">
                        <div style="background:#ddd;height:6px;border-radius:3px;overflow:hidden;">
                            <div style="background:${isWarning ? '#ff5722' : '#ff9800'};height:100%;width:${percentComplete}%;transition:width 0.3s;"></div>
                        </div>
                    </div>
                </div>
                <div style="text-align:right;font-size:12px;color:#666;">${remaining}m left</div>
            </div>`;
    });

    container.innerHTML = html;
}

function updateStatistics() {
    const missingCount = document.querySelectorAll('.student-row .status.missing').length;
    const trackedCount = missingStudentTimestamps.size;
    const lockedCount  = lockedAbsentStudents.size;

    const el = (id) => document.getElementById(id);
    if (el('statMissing'))    el('statMissing').textContent    = missingCount;
    if (el('statTracked'))    el('statTracked').textContent    = trackedCount;
    if (el('statAutoAbsent')) el('statAutoAbsent').textContent = lockedCount;
}


// ============================================
// INTEGRATION HOOKS
// ============================================

/**
 * Called inside updateStudentStatuses for every incoming status from backend.
 * Returns false to BLOCK a status change, true to ALLOW it.
 */
function enhanceUpdateStudentStatuses_MissingTracking(studentId, newStatus, currentStatus) {
    // Block if student is locked absent in memory
    if (lockedAbsentStudents.has(studentId) && newStatus !== 'absent') {
        console.log(`🔒 BLOCKED backend override for locked student ${studentId}`);
        return false;
    }

    // Track new missing students
    if (newStatus === 'missing' && !missingStudentTimestamps.has(studentId)) {
        missingStudentTimestamps.set(studentId, { timestamp: new Date() });
        updateTrackedMissingDisplay();
        updateStatistics();
    }
    // Remove from tracking if no longer missing and not locked
    else if (newStatus !== 'missing' && missingStudentTimestamps.has(studentId)) {
        if (!lockedAbsentStudents.has(studentId)) {
            missingStudentTimestamps.delete(studentId);
            updateTrackedMissingDisplay();
            updateStatistics();
        }
    }

    return true;
}

/**
 * Called when session ends — clear all tracking state
 */
function enhanceConfirmEndSession_MissingTracking() {
    missingStudentTimestamps.clear();
    lockedAbsentStudents.clear();
    stopMissingMonitoring();
    console.log('Cleared missing tracking and locked-absent set');
}


// ============================================
// INITIALIZATION
// ============================================

function initializeMissingThresholdSystem() {
    console.log('🚀 Initializing Missing Threshold System...');
    loadMissingThresholdSettings();
    if (window.missingThresholdConfig.enabled) startMissingMonitoring();

    setInterval(() => {
        updateStatistics();
        updateTrackedMissingDisplay();
    }, 5000);

    console.log('✅ Missing Threshold System ready');
}

function loadMissingThresholdSettings() {
    try {
        const saved = localStorage.getItem('missingThresholdConfig');
        if (saved) {
            Object.assign(window.missingThresholdConfig, JSON.parse(saved));
            const slider = document.getElementById('missingThresholdSlider');
            if (slider) slider.value = window.missingThresholdConfig.thresholdMinutes;
            updateMissingThresholdDisplay(window.missingThresholdConfig.thresholdMinutes);
        }
    } catch (err) { console.error('Error loading settings:', err); }
}

function saveMissingThresholdSettings() {
    try { localStorage.setItem('missingThresholdConfig', JSON.stringify(window.missingThresholdConfig)); }
    catch (err) { console.error('Error saving settings:', err); }
}


// ============================================
// DEBUG HELPERS
// ============================================
window.missingThresholdDebug = {
    getConfig:    () => window.missingThresholdConfig,
    getTracked:   () => missingStudentTimestamps,
    getLocked:    () => lockedAbsentStudents,
    forceCheck:   checkMissingThresholdViolations,
    simulateViolation: (studentId) => {
        const pastTime = new Date(Date.now() - (window.missingThresholdConfig.thresholdMinutes + 1) * 60 * 1000);
        missingStudentTimestamps.set(studentId, { timestamp: pastTime });
        checkMissingThresholdViolations();
    },
    // Manually unlock a student (for testing)
    unlock: (studentId) => {
        lockedAbsentStudents.delete(studentId);
        const row = document.querySelector(`.student-row[data-student-id="${studentId}"]`);
        if (row) { row.removeAttribute('data-locked'); row.removeAttribute('data-lock-reason'); }
        console.log(`🔓 Unlocked ${studentId}`);
    }
};

console.log('📦 Missing Threshold System (FIXED) loaded');
console.log('   Use window.missingThresholdDebug for testing');