const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const path = require('path');

const app = express();
app.use(cors());
app.use(express.json({ limit: '50mb' })); 
app.use(express.static(path.join(__dirname, 'public'))); // Tells Node to serve the 'public' folder

// --- MONGODB CONNECTION (UPDATED FOR LATEST MONGOOSE) ---
mongoose.connect('mongodb://127.0.0.1:27017/retinaguard')
  .then(() => console.log("✅ MongoDB Connected Successfully"))
  .catch(err => console.log("❌ MongoDB Error:", err));

// --- DATABASE SCHEMA ---
const ReportSchema = new mongoose.Schema({
    patientId: String,
    date: { type: Date, default: Date.now },
    results: {
        vessels: String,
        macula: String,
        drusen: String,
        nerves: String,
        bleeds: String,
        scar: String,
    },
    verdict: String,
    confidence: String
});

const Report = mongoose.model('Report', ReportSchema);

// --- API ROUTES ---

// Save a new patient report
app.post('/api/reports', async (req, res) => {
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
    try {
        const reports = await Report.find().sort({ date: -1 }); // Newest first
        res.status(200).json(reports);
    } catch (error) {
        res.status(500).json({ error: "Failed to fetch reports" });
    }
});

// Serve the Frontend Application (Updated routing catch)
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

const PORT = 5000;
app.listen(PORT, () => console.log(`🚀 Server running on http://localhost:${PORT}`));