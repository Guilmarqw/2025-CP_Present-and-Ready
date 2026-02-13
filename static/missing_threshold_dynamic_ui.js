/**
 * DYNAMIC MISSING THRESHOLD SYSTEM
 * 
 * When a student is marked as "MISSING" and exceeds the missing threshold,
 * they are automatically changed to "ABSENT" and locked (cannot change back even if detected)
 */

// ============================================
// GLOBAL STATE FOR MISSING THRESHOLD
// ============================================
window.missingThresholdConfig = {
    enabled: true,                    // Enable/disable missing threshold monitoring
    thresholdMinutes: 1,              // How long before missing becomes absent (default 1 minute)
    checkIntervalSeconds: 15,         // Check every 15 seconds
    lockAbsentStatus: true,           // Lock status as absent (prevent change back)
    notifyOnChange: true              // Show notification when status changes
};

// Track when students were marked as missing
let missingStudentTimestamps = new Map(); // studentId -> { timestamp, locked }

// Track students who are locked as absent
let lockedAbsentStudents = new Set(); // Set of student IDs that are locked

// Interval ID for cleanup
let missingCheckIntervalId = null;


// ============================================
// ADD TO YOUR SETTINGS MODAL HTML
// Insert this inside your existing settingsModal's modal-body
// ============================================

const missingThresholdSettingsHTML = `
<!-- Missing Threshold Configuration Section -->
<div class="settings-section" style="border-top: 2px solid #e0e0e0; margin-top: 20px; padding-top: 20px;">
    <div class="section-header" style="display: flex; align-items: center; gap: 10px; margin-bottom: 15px;">
        <span class="material-icons" style="color: #ff9800; font-size: 28px;">person_off</span>
        <h4 style="margin: 0; color: #333;">Missing Threshold Settings</h4>
    </div>
    
    <!-- Enable/Disable Missing Threshold -->
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

    <!-- Current Missing Threshold -->
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
                    Students missing for more than this time will become ABSENT
                </div>
            </div>
            
            <!-- Slider for Minutes -->
            <div style="margin-bottom: 20px;">
                <label style="font-size: 13px; color: #666; margin-bottom: 8px; display: block;">
                    Adjust Threshold (1-30 minutes):
                </label>
                <input type="range" 
                       id="missingThresholdSlider" 
                       min="1" 
                       max="30" 
                       value="1" 
                       step="1"
                       style="width: 100%;"
                       oninput="updateMissingThresholdDisplay(this.value)">
                <div style="display: flex; justify-content: space-between; font-size: 11px; color: #999; margin-top: 5px;">
                    <span>1 min</span>
                    <span>15 min</span>
                    <span>30 min</span>
                </div>
            </div>
            
            <!-- Quick Set Buttons -->
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px;">
                <button class="btn-quick-set" onclick="setMissingThreshold(1)">1 min</button>
                <button class="btn-quick-set" onclick="setMissingThreshold(3)">3 min</button>
                <button class="btn-quick-set" onclick="setMissingThreshold(5)">5 min</button>
                <button class="btn-quick-set" onclick="setMissingThreshold(10)">10 min</button>
            </div>
        </div>
    </div>

    <!-- Check Frequency -->
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

    <!-- Lock Status Option -->
    <div class="form-group" style="background: #ffebee; padding: 15px; border-radius: 8px; border-left: 4px solid #f44336; margin-bottom: 20px;">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div style="flex: 1;">
                <label style="font-weight: 600; margin: 0; display: flex; align-items: center; gap: 8px;">
                    <span class="material-icons" style="color: #f44336;">lock</span>
                    Lock ABSENT Status (Recommended)
                </label>
                <div style="font-size: 12px; color: #666; margin-top: 4px;">
                    Once marked ABSENT from missing, student cannot change back to PRESENT even if detected
                </div>
            </div>
            <label class="switch">
                <input type="checkbox" id="lockAbsentStatus" checked onchange="toggleLockAbsentStatus(this.checked)">
                <span class="slider round"></span>
            </label>
        </div>
    </div>

    <!-- Currently Tracked Missing Students -->
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

    <!-- Statistics -->
    <div class="form-group" style="margin-top: 20px;">
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;">
            <div style="background: #e3f2fd; padding: 12px; border-radius: 8px; text-align: center;">
                <div style="font-size: 11px; color: #666; margin-bottom: 4px;">Missing</div>
                <div style="font-size: 24px; font-weight: bold; color: #2196f3;">
                    <span id="statMissing">0</span>
                </div>
            </div>
            <div style="background: #fff3e0; padding: 12px; border-radius: 8px; text-align: center;">
                <div style="font-size: 11px; color: #666; margin-bottom: 4px;">Tracked</div>
                <div style="font-size: 24px; font-weight: bold; color: #ff9800;">
                    <span id="statTracked">0</span>
                </div>
            </div>
            <div style="background: #ffebee; padding: 12px; border-radius: 8px; text-align: center;">
                <div style="font-size: 11px; color: #666; margin-bottom: 4px;">Auto-Absent</div>
                <div style="font-size: 24px; font-weight: bold; color: #f44336;">
                    <span id="statAutoAbsent">0</span>
                </div>
            </div>
        </div>
    </div>
</div>
`;

// ============================================
// ADD CSS STYLES FOR THE UI
// ============================================

const missingThresholdStyles = `
<style>
/* Toggle Switch Styles */
.switch {
    position: relative;
    display: inline-block;
    width: 50px;
    height: 24px;
}

.switch input {
    opacity: 0;
    width: 0;
    height: 0;
}

.slider {
    position: absolute;
    cursor: pointer;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: #ccc;
    transition: .4s;
}

.slider:before {
    position: absolute;
    content: "";
    height: 16px;
    width: 16px;
    left: 4px;
    bottom: 4px;
    background-color: white;
    transition: .4s;
}

input:checked + .slider {
    background-color: #4CAF50;
}

input:checked + .slider:before {
    transform: translateX(26px);
}

.slider.round {
    border-radius: 24px;
}

.slider.round:before {
    border-radius: 50%;
}

/* Quick Set Buttons */
.btn-quick-set {
    padding: 8px 12px;
    background: white;
    border: 2px solid #ff9800;
    color: #ff9800;
    border-radius: 6px;
    cursor: pointer;
    font-weight: 600;
    font-size: 13px;
    transition: all 0.3s;
}

.btn-quick-set:hover {
    background: #ff9800;
    color: white;
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(255, 152, 0, 0.3);
}

.btn-quick-set:active {
    transform: translateY(0);
}

/* Range Slider Styling */
input[type="range"] {
    -webkit-appearance: none;
    appearance: none;
    height: 8px;
    background: #ddd;
    border-radius: 5px;
    outline: none;
}

input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 20px;
    height: 20px;
    background: #ff9800;
    cursor: pointer;
    border-radius: 50%;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}

input[type="range"]::-moz-range-thumb {
    width: 20px;
    height: 20px;
    background: #ff9800;
    cursor: pointer;
    border-radius: 50%;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    border: none;
}

/* Form Select */
.form-select {
    width: 100%;
    padding: 10px 12px;
    border: 1px solid #ddd;
    border-radius: 6px;
    font-size: 14px;
    background: white;
    cursor: pointer;
}

.form-select:focus {
    outline: none;
    border-color: #ff9800;
    box-shadow: 0 0 0 3px rgba(255, 152, 0, 0.1);
}

/* Missing Student Card */
.missing-student-card {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px;
    background: white;
    border-left: 4px solid #ff9800;
    border-radius: 6px;
    margin-bottom: 8px;
}

.missing-student-card.warning {
    border-left-color: #ff5722;
    background: #fff3e0;
}

.missing-student-card.locked {
    border-left-color: #f44336;
    background: #ffebee;
}
</style>
`;

// Add styles to document
document.head.insertAdjacentHTML('beforeend', missingThresholdStyles);


// ============================================
// UI CONTROL FUNCTIONS
// ============================================

/**
 * Update missing threshold display when slider changes
 */
function updateMissingThresholdDisplay(minutes) {
    const display = document.getElementById('currentMissingThreshold');
    const plural = document.getElementById('missingThresholdPlural');
    
    if (display) {
        display.textContent = minutes;
    }
    
    if (plural) {
        plural.textContent = minutes == 1 ? '' : 's';
    }
    
    // Update config (but don't apply yet)
    window.missingThresholdConfig.thresholdMinutes = parseInt(minutes);
    
    console.log(`📝 Missing threshold updated to ${minutes} minute(s) (not applied yet)`);
}

/**
 * Set missing threshold to specific value
 */
function setMissingThreshold(minutes) {
    const slider = document.getElementById('missingThresholdSlider');
    
    if (slider) {
        slider.value = minutes;
    }
    
    updateMissingThresholdDisplay(minutes);
    
    // Show feedback
    showModalNotification(`Missing threshold set to ${minutes} minute${minutes == 1 ? '' : 's'}`, 'success', 2000);
}

/**
 * Apply missing threshold settings (call this when user clicks "Save" in settings modal)
 */
async function applyMissingThresholdSettings() {
    const threshold = window.missingThresholdConfig.thresholdMinutes;
    const sessionId = window._sessionData?.session_id;
    
    if (!sessionId) {
        showModalNotification('Session ID not found', 'error', 3000);
        return false;
    }
    
    try {
        // Send to backend
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
            console.log(`✅ Missing threshold applied: ${threshold} minute(s)`);
            showModalNotification(`Missing threshold set to ${threshold} minute${threshold == 1 ? '' : 's'}`, 'success', 3000);
            
            // Restart monitoring with new settings
            restartMissingMonitoring();
            
            return true;
        } else {
            showModalNotification(data.message || 'Failed to update threshold', 'error', 3000);
            return false;
        }
    } catch (error) {
        console.error('Error applying missing threshold:', error);
        showModalNotification('Network error updating threshold', 'error', 3000);
        return false;
    }
}

/**
 * Toggle missing threshold monitoring on/off
 */
function toggleMissingThreshold(enabled) {
    window.missingThresholdConfig.enabled = enabled;
    
    if (enabled) {
        console.log('✅ Missing threshold monitoring ENABLED');
        startMissingMonitoring();
        showModalNotification('Missing threshold monitoring enabled', 'success', 2000);
    } else {
        console.log('❌ Missing threshold monitoring DISABLED');
        stopMissingMonitoring();
        showModalNotification('Missing threshold monitoring disabled', 'warning', 2000);
    }
}

/**
 * Toggle lock absent status
 */
function toggleLockAbsentStatus(enabled) {
    window.missingThresholdConfig.lockAbsentStatus = enabled;
    
    if (enabled) {
        console.log('🔒 Lock absent status ENABLED - Students cannot change back from ABSENT');
        showModalNotification('Lock absent status enabled', 'success', 2000);
    } else {
        console.log('🔓 Lock absent status DISABLED - Students can change from ABSENT');
        showModalNotification('Lock absent status disabled', 'warning', 2000);
    }
}

/**
 * Update check frequency
 */
function updateCheckFrequency(seconds) {
    window.missingThresholdConfig.checkIntervalSeconds = parseInt(seconds);
    
    console.log(`⏱️ Check frequency updated to every ${seconds} seconds`);
    
    // Restart monitoring with new frequency
    if (window.missingThresholdConfig.enabled) {
        restartMissingMonitoring();
    }
    
    showModalNotification(`Check frequency set to every ${seconds} seconds`, 'success', 2000);
}


// ============================================
// CORE MONITORING FUNCTIONS
// ============================================

/**
 * Start missing threshold monitoring
 */
function startMissingMonitoring() {
    if (missingCheckIntervalId) {
        clearInterval(missingCheckIntervalId);
    }
    
    const intervalMs = window.missingThresholdConfig.checkIntervalSeconds * 1000;
    
    missingCheckIntervalId = setInterval(() => {
        if (isSessionEnding || isDetectionStopped) return;
        checkMissingThresholdViolations();
    }, intervalMs);
    
    console.log(`🚀 Missing monitoring started (checking every ${window.missingThresholdConfig.checkIntervalSeconds}s)`);
}

/**
 * Stop missing threshold monitoring
 */
function stopMissingMonitoring() {
    if (missingCheckIntervalId) {
        clearInterval(missingCheckIntervalId);
        missingCheckIntervalId = null;
    }
    
    console.log('⏹️ Missing monitoring stopped');
}

/**
 * Restart monitoring (when settings change)
 */
function restartMissingMonitoring() {
    stopMissingMonitoring();
    
    if (window.missingThresholdConfig.enabled) {
        startMissingMonitoring();
    }
}

/**
 * Check for missing threshold violations
 * This is the MAIN function that runs periodically
 */
async function checkMissingThresholdViolations() {
    if (!window.missingThresholdConfig.enabled) return;
    
    console.log('🔍 Checking for missing threshold violations...');
    
    const thresholdMinutes = window.missingThresholdConfig.thresholdMinutes;
    const thresholdMs = thresholdMinutes * 60 * 1000;
    const now = new Date();
    
    const studentRows = document.querySelectorAll('.student-row');
    let violationsFound = 0;
    
    for (const row of studentRows) {
        const statusElement = row.querySelector('.status');
        const studentId = row.getAttribute('data-student-id');
        const studentName = row.querySelector('.student-name')?.textContent;
        
        if (!statusElement || !studentId) continue;
        
        const currentStatus = statusElement.textContent.toLowerCase();
        
        // Check if student is locked as absent
        if (lockedAbsentStudents.has(studentId)) {
            // Prevent any status change - keep as absent
            if (currentStatus !== 'absent') {
                console.log(`🔒 LOCKED: ${studentName} - Forcing back to ABSENT`);
                forceStudentAbsent(studentId, studentName, 'Student is locked as absent');
            }
            continue;
        }
        
        // Track when student becomes missing
        if (currentStatus === 'missing') {
            if (!missingStudentTimestamps.has(studentId)) {
                missingStudentTimestamps.set(studentId, { 
                    timestamp: now, 
                    locked: false 
                });
                console.log(`📝 Started tracking missing student: ${studentName} (ID: ${studentId})`);
                updateTrackedMissingDisplay();
            }
            
            // Calculate how long they've been missing
            const trackData = missingStudentTimestamps.get(studentId);
            const missingTime = trackData.timestamp;
            const elapsedMs = now - missingTime;
            const elapsedMinutes = Math.floor(elapsedMs / 60000);
            const elapsedSeconds = Math.floor(elapsedMs / 1000);
            
            console.log(`  ⏱️ ${studentName}: Missing for ${elapsedMinutes}m ${elapsedSeconds % 60}s (threshold: ${thresholdMinutes}m)`);
            
            // Check if threshold exceeded
            if (elapsedMs > thresholdMs) {
                console.log(`  ⚠️ THRESHOLD EXCEEDED: ${studentName} (${elapsedMinutes}m > ${thresholdMinutes}m)`);
                
                await changeStudentToAbsentFromMissing(
                    studentId, 
                    studentName, 
                    `Missing for ${elapsedMinutes} minute${elapsedMinutes !== 1 ? 's' : ''}`
                );
                
                violationsFound++;
            }
        } 
        // Clean up tracking for students no longer missing
        else if (missingStudentTimestamps.has(studentId)) {
            if (currentStatus === 'present') {
                console.log(`  ✅ ${studentName} returned - removing from missing tracking`);
                missingStudentTimestamps.delete(studentId);
            } else if (currentStatus === 'absent') {
                console.log(`  ❌ ${studentName} marked absent - removing from missing tracking`);
                missingStudentTimestamps.delete(studentId);
            } else if (currentStatus === 'late') {
                console.log(`  🕐 ${studentName} marked late - removing from missing tracking`);
                missingStudentTimestamps.delete(studentId);
            }
            updateTrackedMissingDisplay();
        }
    }
    
    if (violationsFound > 0) {
        console.log(`  🚨 Found ${violationsFound} missing threshold violation(s)`);
        updateStudentCount();
        updateStatistics();
    } else {
        console.log('  ✅ No missing threshold violations found');
    }
    
    updateTrackedMissingDisplay();
}

/**
 * Change student from MISSING to ABSENT (automatic)
 */
async function changeStudentToAbsentFromMissing(studentId, studentName, reason) {
    console.log(`  🤖 AUTO: Changing ${studentId} from MISSING to ABSENT`);
    
    try {
        const action = 'mark_present';
        const student_data = { 
            student_id: studentId, 
            status: 'absent',
            remarks: `Auto-absent: ${reason}`,
            automatic: true,
            from_missing: true,
            locked: window.missingThresholdConfig.lockAbsentStatus
        };
        
        const res = await fetch('/api/manage_student', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action, student_data })
        });
        
        const data = await res.json();
        
        if (data.success) {
            console.log(`  ✅ Successfully changed ${studentId} to ABSENT`);
            
            // Update UI immediately
            const studentRow = document.querySelector(`.student-row[data-student-id="${studentId}"]`);
            if (studentRow) {
                const statusElement = studentRow.querySelector('.status');
                if (statusElement) {
                    statusElement.className = 'status absent';
                    statusElement.textContent = 'Absent';
                }
            }
            
            // Lock the student if enabled
            if (window.missingThresholdConfig.lockAbsentStatus) {
                lockedAbsentStudents.add(studentId);
                console.log(`  🔒 Student ${studentId} is now LOCKED as absent`);
            }
            
            // Remove from missing tracking
            missingStudentTimestamps.delete(studentId);
            
            // Show notification
            showMissingThresholdNotification(studentId, studentName, reason);
            
            updateStatistics();
        } else {
            console.error(`  ❌ Failed to change status: ${data.message}`);
        }
    } catch (err) {
        console.error('Error changing student to absent:', err);
    }
}

/**
 * Force student to stay absent (for locked students)
 */
async function forceStudentAbsent(studentId, studentName, reason) {
    try {
        const action = 'mark_present';
        const student_data = { 
            student_id: studentId, 
            status: 'absent',
            remarks: reason,
            automatic: true,
            locked: true
        };
        
        const res = await fetch('/api/manage_student', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action, student_data })
        });
        
        // Update UI
        const studentRow = document.querySelector(`.student-row[data-student-id="${studentId}"]`);
        if (studentRow) {
            const statusElement = studentRow.querySelector('.status');
            if (statusElement) {
                statusElement.className = 'status absent';
                statusElement.textContent = 'Absent';
            }
        }
    } catch (err) {
        console.error('Error forcing student absent:', err);
    }
}


// ============================================
// UI UPDATE FUNCTIONS
// ============================================

/**
 * Show notification when student is auto-marked absent from missing
 */
function showMissingThresholdNotification(studentId, studentName, reason) {
    const notification = document.createElement('div');
    notification.className = 'threshold-notification missing-threshold-notification';
    notification.innerHTML = `
        <div style="
            display: flex; 
            align-items: center; 
            gap: 12px; 
            padding: 15px 20px; 
            background: #ffebee; 
            border-left: 4px solid #f44336; 
            box-shadow: 0 4px 12px rgba(0,0,0,0.15); 
            border-radius: 8px; 
            margin: 10px; 
            animation: slideIn 0.3s ease-out;
        ">
            <span class="material-icons" style="color: #f44336; font-size: 28px;">person_remove</span>
            <div style="flex: 1;">
                <div style="font-weight: 600; font-size: 15px; color: #333;">
                    <strong>${studentName || studentId}</strong> marked as <strong style="color: #f44336;">ABSENT</strong>
                </div>
                <div style="font-size: 13px; color: #666; margin-top: 3px;">
                    ${reason}
                </div>
                ${window.missingThresholdConfig.lockAbsentStatus ? 
                    '<div style="font-size: 12px; color: #f44336; margin-top: 3px; display: flex; align-items: center; gap: 4px;"><span class="material-icons" style="font-size: 14px;">lock</span> Status locked - cannot change back</div>' 
                    : ''}
            </div>
            <button onclick="this.parentElement.parentElement.remove()" style="
                background: none; 
                border: none; 
                cursor: pointer;
                padding: 4px;
            ">
                <span class="material-icons" style="color: #999;">close</span>
            </button>
        </div>
    `;
    
    const studentList = document.getElementById('studentList');
    if (studentList) {
        studentList.insertBefore(notification, studentList.firstChild);
        
        // Auto-remove after 12 seconds
        setTimeout(() => {
            if (notification.parentElement) {
                notification.style.animation = 'slideOut 0.3s ease-in';
                setTimeout(() => notification.remove(), 300);
            }
        }, 12000);
    }
}

/**
 * Update the tracked missing students display in settings modal
 */
function updateTrackedMissingDisplay() {
    const container = document.getElementById('trackedMissingStudents');
    if (!container) return;
    
    if (missingStudentTimestamps.size === 0) {
        container.innerHTML = `
            <div style="text-align: center; color: #999; font-style: italic; padding: 20px;">
                No students currently marked as missing
            </div>
        `;
        return;
    }
    
    const now = new Date();
    const thresholdMs = window.missingThresholdConfig.thresholdMinutes * 60 * 1000;
    
    let html = '';
    
    missingStudentTimestamps.forEach((trackData, studentId) => {
        const studentRow = document.querySelector(`.student-row[data-student-id="${studentId}"]`);
        const studentName = studentRow?.querySelector('.student-name')?.textContent || studentId;
        
        const elapsedMs = now - trackData.timestamp;
        const elapsedMinutes = Math.floor(elapsedMs / 60000);
        const elapsedSeconds = Math.floor((elapsedMs % 60000) / 1000);
        
        const percentComplete = Math.min((elapsedMs / thresholdMs) * 100, 100);
        const isWarning = percentComplete > 75;
        
        html += `
            <div class="missing-student-card ${isWarning ? 'warning' : ''}">
                <div style="flex: 1;">
                    <div style="font-weight: 600; margin-bottom: 4px;">${studentName}</div>
                    <div style="font-size: 12px; color: #666;">
                        Missing for: ${elapsedMinutes}m ${elapsedSeconds}s
                    </div>
                    <div style="margin-top: 6px;">
                        <div style="background: #ddd; height: 6px; border-radius: 3px; overflow: hidden;">
                            <div style="background: ${isWarning ? '#ff5722' : '#ff9800'}; height: 100%; width: ${percentComplete}%; transition: width 0.3s;"></div>
                        </div>
                    </div>
                </div>
                <div style="text-align: right; font-size: 12px; color: #666;">
                    ${Math.max(0, window.missingThresholdConfig.thresholdMinutes - elapsedMinutes)}m left
                </div>
            </div>
        `;
    });
    
    container.innerHTML = html;
}

/**
 * Update statistics display
 */
function updateStatistics() {
    // Count missing students
    const missingCount = document.querySelectorAll('.student-row .status.missing').length;
    const trackedCount = missingStudentTimestamps.size;
    const lockedCount = lockedAbsentStudents.size;
    
    const missingEl = document.getElementById('statMissing');
    const trackedEl = document.getElementById('statTracked');
    const autoAbsentEl = document.getElementById('statAutoAbsent');
    
    if (missingEl) missingEl.textContent = missingCount;
    if (trackedEl) trackedEl.textContent = trackedCount;
    if (autoAbsentEl) autoAbsentEl.textContent = lockedCount;
}


// ============================================
// INTEGRATION WITH EXISTING FUNCTIONS
// ============================================

/**
 * Enhance your existing updateStudentStatuses function
 * Add this code inside the status change detection
 */
function enhanceUpdateStudentStatuses_MissingTracking(studentId, newStatus, currentStatus) {
    // Track when student becomes missing
    if (newStatus === 'missing' && !missingStudentTimestamps.has(studentId)) {
        missingStudentTimestamps.set(studentId, { 
            timestamp: new Date(), 
            locked: false 
        });
        console.log(`  📝 Student ${studentId} marked as missing - starting threshold timer`);
        updateTrackedMissingDisplay();
        updateStatistics();
    }
    // Remove from tracking if no longer missing (unless locked)
    else if (newStatus !== 'missing' && missingStudentTimestamps.has(studentId)) {
        if (!lockedAbsentStudents.has(studentId)) {
            missingStudentTimestamps.delete(studentId);
            console.log(`  ✅ Student ${studentId} status changed to ${newStatus} - removed from missing tracking`);
            updateTrackedMissingDisplay();
            updateStatistics();
        }
    }
    
    // Prevent locked students from changing status
    if (lockedAbsentStudents.has(studentId) && newStatus !== 'absent') {
        console.log(`  🔒 BLOCKED: Student ${studentId} is locked as absent - preventing status change`);
        return false; // Block the status change
    }
    
    return true; // Allow the status change
}

/**
 * Add to your existing confirmEndSession function
 */
function enhanceConfirmEndSession_MissingTracking() {
    // Clear all tracking
    missingStudentTimestamps.clear();
    lockedAbsentStudents.clear();
    stopMissingMonitoring();
    
    console.log('  Cleared missing student tracking and locked students');
}


// ============================================
// INITIALIZATION
// ============================================

/**
 * Initialize missing threshold system
 * Call this when page loads (add to your window.addEventListener('load', ...))
 */
function initializeMissingThresholdSystem() {
    console.log('🚀 Initializing Missing Threshold System...');
    
    // Load settings from session or localStorage
    loadMissingThresholdSettings();
    
    // Start monitoring if enabled
    if (window.missingThresholdConfig.enabled) {
        startMissingMonitoring();
    }
    
    // Update statistics every 5 seconds
    setInterval(() => {
        updateStatistics();
        updateTrackedMissingDisplay();
    }, 5000);
    
    console.log('✅ Missing Threshold System initialized');
    console.log('   Threshold:', window.missingThresholdConfig.thresholdMinutes, 'minute(s)');
    console.log('   Check interval:', window.missingThresholdConfig.checkIntervalSeconds, 'second(s)');
    console.log('   Lock absent:', window.missingThresholdConfig.lockAbsentStatus);
}

/**
 * Load settings (from localStorage or session)
 */
function loadMissingThresholdSettings() {
    try {
        const saved = localStorage.getItem('missingThresholdConfig');
        if (saved) {
            const config = JSON.parse(saved);
            Object.assign(window.missingThresholdConfig, config);
            
            // Update UI
            const slider = document.getElementById('missingThresholdSlider');
            if (slider) slider.value = config.thresholdMinutes;
            
            updateMissingThresholdDisplay(config.thresholdMinutes);
            
            console.log('📂 Loaded saved missing threshold settings');
        }
    } catch (err) {
        console.error('Error loading settings:', err);
    }
}

/**
 * Save settings to localStorage
 */
function saveMissingThresholdSettings() {
    try {
        localStorage.setItem('missingThresholdConfig', JSON.stringify(window.missingThresholdConfig));
        console.log('💾 Missing threshold settings saved');
    } catch (err) {
        console.error('Error saving settings:', err);
    }
}


// ============================================
// EXPORT FOR TESTING
// ============================================

window.missingThresholdDebug = {
    getConfig: () => window.missingThresholdConfig,
    getTracked: () => missingStudentTimestamps,
    getLocked: () => lockedAbsentStudents,
    forceCheck: checkMissingThresholdViolations,
    simulateViolation: (studentId) => {
        const pastTime = new Date(Date.now() - (window.missingThresholdConfig.thresholdMinutes + 1) * 60 * 1000);
        missingStudentTimestamps.set(studentId, { timestamp: pastTime, locked: false });
        console.log(`Simulated violation for ${studentId}`);
        checkMissingThresholdViolations();
    }
};


console.log('📦 Missing Threshold System loaded');
console.log('   Use window.missingThresholdDebug for testing');