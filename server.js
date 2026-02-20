const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const path = require('path');

const app = express();
app.use(cors());
app.use(express.json({ limit: '50mb' })); 
app.use(express.static(path.join(__dirname, 'public'))); // Tells Node to serve the 'public' folder

// --- MONGODB CONNECTION (OPTIONAL - SYSTEM WORKS WITHOUT IT) ---
let mongoConnected = false;
mongoose.connect('mongodb://127.0.0.1:27017/retinaguard')
  .then(() => {
    console.log("✅ MongoDB Connected Successfully");
    mongoConnected = true;
  })
  .catch(err => {
    console.log("⚠️  MongoDB Not Available (Report saving disabled)");
    console.log("   → Image analysis will still work normally");
    console.log("   → To enable database: Install and start MongoDB on port 27017");
  });

// --- DATABASE SCHEMA ---
const ReportSchema = new mongoose.Schema({
    patientId: String,
    date: { type: Date, default: Date.now },
    
    // Patient Demographics & History
    patient_history: {
        age: Number,
        ethnicity: String,
        symptoms: Object,
        family_history: Boolean,
        risk_score: Number
    },
    
    // Image Metadata
    camera_type: String,
    quality_score: Number,
    is_angiography: Boolean,
    angiography_warning: String,
    
    // Expert Analysis Results
    results: Object, // All expert opinions
    expert_opinions: Array, // Full expert array
    
    // Diagnosis & Findings
    verdict: String,
    verdict_code: String,
    confidence: String,
    compositeScore: Number,
    
    // Clinical Details
    triadStatus: Object,
    criticalFindings: Array,
    differential_diagnosis: Array,
    
    // Metadata
    image_resolution: String,
    analysis_duration: Number
});

const Report = mongoose.model('Report', ReportSchema);

// --- API ROUTES ---

// Save a new patient report
app.post('/api/reports', async (req, res) => {
    if (!mongoConnected) {
        return res.status(503).json({ 
            error: "Database unavailable", 
            message: "Report saving is disabled (MongoDB not running). Analysis results are still available on screen." 
        });
    }
    try {
        const newReport = new Report(req.body);
        await newReport.save();
        res.status(201).json({ message: "Report saved successfully!", id: newReport._id });
    } catch (error) {
        console.error(error);
        res.status(500).json({ error: "Failed to save report" });
    }
});

// Fetch all patient records
app.get('/api/reports', async (req, res) => {
    if (!mongoConnected) {
        return res.status(503).json({ 
            error: "Database unavailable", 
            message: "MongoDB is not running. No historical reports available." 
        });
    }
    try {
        const reports = await Report.find().sort({ date: -1 }); // Newest first
        res.status(200).json(reports);
    } catch (error) {
        res.status(500).json({ error: "Failed to fetch reports" });
    }
});

// Fetch individual report by ID
app.get('/api/reports/:id', async (req, res) => {
    if (!mongoConnected) {
        return res.status(503).json({ 
            error: "Database unavailable", 
            message: "MongoDB is not running." 
        });
    }
    try {
        const report = await Report.findById(req.params.id);
        if (!report) {
            return res.status(404).json({ error: "Report not found" });
        }
        res.status(200).json(report);
    } catch (error) {
        res.status(500).json({ error: "Failed to fetch report" });
    }
});

// Serve the Frontend Application (Updated routing catch)
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

const PORT = 5000;
app.listen(PORT, () => console.log(`🚀 Server running on http://localhost:${PORT}`));