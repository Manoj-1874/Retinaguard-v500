/**
 * ================================================================================
 * RETINAGUARD V500 - FRONTEND INTEGRATION SCRIPT
 * ================================================================================
 * Connects the UI to Flask AI Server for real-time RP diagnosis
 */

// API Configuration
const FLASK_API_URL = 'http://localhost:5001';
const NODE_API_URL = 'http://localhost:5000';

// Global state for current diagnosis
let currentDiagnosisResults = null;

/**
 * Call Flask API to analyze retinal image
 * @param {string} base64Image - Base64 encoded image data
 * @param {string} patientId - Patient identifier
 * @returns {Promise<Object>} Diagnosis results
 */
async function analyzeWithFlaskAPI(base64Image, patientId) {
    try {
        console.log('📡 Sending image to Flask API for analysis...');
        
        const response = await fetch(`${FLASK_API_URL}/api/analyze`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image: base64Image,
                patientId: patientId
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log('✅ Analysis complete:', data);
        
        return data;
        
    } catch (error) {
        console.error('❌ Flask API Error:', error);
        throw new Error(`Cannot connect to Flask AI Server. Please ensure it's running on ${FLASK_API_URL}`);
    }
}

/**
 * Update UI with diagnosis results from Flask
 * @param {Object} results - Diagnosis results from Flask API
 */
function updateUIWithResults(results) {
    const crates = document.querySelectorAll('.crate-card');
    const expertMapping = {
        'ai_pattern': 'ai_pattern',
        'vessels': 'vessels',
        'pigment': 'pigment',
        'optic_disc': 'optic_disc',
        'tortuosity': 'tortuosity',
        'texture': 'texture',
        'spatial': 'spatial'
    };

    // Update each expert card
    crates.forEach((crate, index) => {
        const expertKey = crate.getAttribute('data-expert');
        const expertData = results.results[expertMapping[expertKey]];
        
        if (!expertData) {
            console.warn(`No data for expert: ${expertKey}`);
            return;
        }

        // Update status indicator
        const statusIndicator = crate.querySelector('.status-indicator');
        const crateIcon = crate.querySelector('.crate-icon');
        
        statusIndicator.textContent = expertData.status;
        
        // Color code by severity
        const severityColors = {
            'CRITICAL': { text: '#ff0055', bg: 'rgba(255,0,85,0.1)' },
            'MODERATE': { text: '#ff9900', bg: 'rgba(255,153,0,0.1)' },
            'MILD': { text: '#fbc02d', bg: 'rgba(251,192,45,0.1)' },
            'NORMAL': { text: '#00ff88', bg: 'rgba(0,255,136,0.1)' }
        };
        
        const colors = severityColors[expertData.severity] || severityColors.NORMAL;
        statusIndicator.style.color = colors.text;
        statusIndicator.style.background = colors.bg;
        
        // Update icon color for critical/moderate
        if (expertData.severity === 'CRITICAL' || expertData.severity === 'MODERATE') {
            crateIcon.style.color = colors.text;
        }
        
        // Add click handler for modal
        crate.onclick = () => {
            openExpertModal(
                crate.querySelector('.crate-label').textContent,
                crateIcon.className.split(' ')[1],
                expertData.confidence,
                expertData.status,
                colors.text,
                expertData
            );
        };
    });

    // Update RP Triad Status
    updateTriadStatus(results.triad_status, results.triad_complete);
}

/**
 * Update RP Triad status panel
 */
function updateTriadStatus(triadStatus, isComplete) {
    document.getElementById('triad-1-status').innerHTML = 
        (triadStatus.triad_2_vessels ? '✅' : '❌') + ' Bone Spicules';
    document.getElementById('triad-2-status').innerHTML = 
        (triadStatus.triad_1_pigment ? '✅' : '❌') + ' Vessel Attenuation';
    document.getElementById('triad-3-status').innerHTML = 
        (triadStatus.triad_3_optic_disc ? '✅' : '❌') + ' Optic Disc Pallor';
    
    // Show triad status panel
    const triadPanel = document.getElementById('triad-status');
    triadPanel.style.opacity = 1;
    
    // Highlight if complete
    if (isComplete) {
        triadPanel.style.borderColor = '#00ff88';
        triadPanel.style.boxShadow = '0 0 30px rgba(0,255,136,0.3)';
    }
}

/**
 * Open detailed expert modal
 */
function openExpertModal(title, iconClass, confidence, status, color, expertData) {
    if (isAnimating) return;
    sfxClick.play().catch(e=>{});
    
    const modal = document.getElementById('detail-modal');
    const box = document.getElementById('modal-box');
    
    document.getElementById('m-title').innerText = title + " ANALYSIS";
    document.getElementById('m-icon').className = `fa-solid ${iconClass} modal-icon-lg`;
    document.getElementById('m-icon').style.color = color;
    document.getElementById('m-conf').innerText = confidence + '%';
    document.getElementById('m-status').innerText = status;
    document.getElementById('m-status').style.color = color;
    document.getElementById('m-graph').style.width = confidence + '%';
    document.getElementById('m-graph').style.background = `linear-gradient(90deg, rgba(0,243,255,0.2), ${color})`;
    
    modal.style.display = 'flex';
    anime({ targets: modal, opacity: 1, duration: 300 });
    anime({ targets: box, scale: [0.9, 1], opacity: [0, 1], duration: 400, easing: 'easeOutBack' });
}

/**
 * Save diagnosis results to MongoDB
 */
async function saveDiagnosisToDatabase(results, patientId) {
    try {
        const reportData = {
            patientId: patientId,
            results: {
                ai_pattern: results.results.ai_pattern.status,
                vessels: results.results.vessels.status,
                pigment: results.results.pigment.status,
                optic_disc: results.results.optic_disc.status,
                tortuosity: results.results.tortuosity.status,
                texture: results.results.texture.status,
                spatial: results.results.spatial.status
            },
            verdict: results.verdict,
            confidence: results.confidence,
            composite_score: results.composite_score,
            triad_complete: results.triad_complete
        };

        const response = await fetch(`${NODE_API_URL}/api/reports`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(reportData)
        });

        if (!response.ok) {
            throw new Error('Database save failed');
        }

        return await response.json();
        
    } catch (error) {
        console.error('❌ Database Error:', error);
        throw new Error(`Cannot save to database. Please ensure Node.js server is running on ${NODE_API_URL}`);
    }
}

/**
 * Check Flask API health
 */
async function checkFlaskHealth() {
    try {
        const response = await fetch(`${FLASK_API_URL}/api/health`);
        if (response.ok) {
            const data = await response.json();
            console.log('✅ Flask Server:', data);
            return true;
        }
        return false;
    } catch (error) {
        console.warn('⚠️ Flask server not responding');
        return false;
    }
}

// Export functions for use in main script
if (typeof window !== 'undefined') {
    window.FlaskIntegration = {
        analyzeWithFlaskAPI,
        updateUIWithResults,
        updateTriadStatus,
        saveDiagnosisToDatabase,
        checkFlaskHealth,
        openExpertModal
    };
}
