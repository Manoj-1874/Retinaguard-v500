import urllib.request
import urllib.parse
import os

dot_code = """
digraph G {
    bgcolor="white";
    rankdir=TB;
    nodesep=0.8;
    ranksep=0.6;
    dpi=300;
    
    node [shape=box, style="rounded,filled", fillcolor="white", color="black", penwidth=3, fontname="Helvetica-Bold", fontsize=28, margin="0.4,0.2"];
    edge [fontname="Helvetica-Bold", fontsize=24, color="black", penwidth=3];
    
    Start [shape=oval, label="Start"];
    Input [label="Upload Retinal Scan"];
    Preprocess [label="Image Quality Validator"];
    QualityCheck [shape=diamond, label="Quality OK?", margin="0.1,0.1"];
    Reject [label="Reject Scan"];
    CNN [label="ResNet50V2\\nFeature Extractor"];
    Topo [label="Geometric\\nFragmentation Filter"];
    Experts [label="10-Expert\\nClinical Panel"];
    RuleEngine [label="6-Rule\\nDecision Engine"];
    DiffCheck [shape=diamond, label="Alternative\\nDisease?", margin="0.1,0.1"];
    DiffOverride [label="Rule 0:\\nDifferential Override"];
    RPCheck [shape=diamond, label="RP Triad\\nPresent?", margin="0.1,0.1"];
    RPPrediction [label="Diagnose:\\nRetinitis Pigmentosa"];
    VetoCheck [shape=diamond, label="AI High but\\n0 Experts?", margin="0.1,0.1"];
    Veto [label="Rule 8:\\nAI Override Healthy"];
    Healthy [label="Diagnose:\\nHealthy"];
    FinalReport [label="Generate Heatmap &\\nTreatment Plan"];
    UI [label="Render 3D UI"];
    End [shape=oval, label="End"];

    Start -> Input;
    Input -> Preprocess;
    Preprocess -> QualityCheck;
    QualityCheck -> Reject [label="No"];
    QualityCheck -> CNN [label="Yes"];
    CNN -> Topo;
    Topo -> Experts;
    Experts -> RuleEngine;
    RuleEngine -> DiffCheck;
    DiffCheck -> DiffOverride [label="Yes"];
    DiffOverride -> FinalReport;
    DiffCheck -> RPCheck [label="No"];
    RPCheck -> RPPrediction [label="Yes"];
    RPPrediction -> FinalReport;
    RPCheck -> VetoCheck [label="No"];
    VetoCheck -> Veto [label="Yes"];
    Veto -> FinalReport;
    VetoCheck -> Healthy [label="No"];
    Healthy -> FinalReport;
    FinalReport -> UI;
    UI -> End;
    
    { rank=same; QualityCheck; Reject; }
    { rank=same; DiffCheck; DiffOverride; }
    { rank=same; RPCheck; RPPrediction; }
    { rank=same; VetoCheck; Veto; }
}
"""

encoded = urllib.parse.quote(dot_code)
url = f"https://quickchart.io/graphviz?graph={encoded}&format=png"

req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
out_path = 'e:/V500/Architecture_Diagram_V500.png'
with urllib.request.urlopen(req) as response, open(out_path, 'wb') as out_file:
    out_file.write(response.read())

print("Downloaded Architecture_Diagram_V500.png successfully!")
