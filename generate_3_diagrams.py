import urllib.request
import urllib.parse

parts = [
    ("Architecture_Part1.png", """digraph G1 {
    bgcolor="white"; rankdir=TB; nodesep=0.8; ranksep=0.6;
    node [shape=box, style="rounded,filled", fillcolor="white", color="black", penwidth=3, fontname="Helvetica-Bold", fontsize=28, margin="0.4,0.2"];
    edge [fontname="Helvetica-Bold", fontsize=24, color="black", penwidth=3];
    Start [shape=oval, label="Start"];
    Input [label="Upload Retinal Scan"];
    Preprocess [label="Image Quality Validator"];
    QualityCheck [shape=diamond, label="Quality OK?", margin="0.1,0.1"];
    Reject [label="Reject Scan"];
    CNN [label="ResNet50V2\\nFeature Extractor"];
    Start -> Input -> Preprocess -> QualityCheck;
    QualityCheck -> Reject [label="No"];
    QualityCheck -> CNN [label="Yes"];
    { rank=same; QualityCheck; Reject; }
}"""),

    ("Architecture_Part2.png", """digraph G2 {
    bgcolor="white"; rankdir=TB; nodesep=0.8; ranksep=0.6;
    node [shape=box, style="rounded,filled", fillcolor="white", color="black", penwidth=3, fontname="Helvetica-Bold", fontsize=28, margin="0.4,0.2"];
    edge [fontname="Helvetica-Bold", fontsize=24, color="black", penwidth=3];
    CNNIn [label="From Feature Extractor", shape=oval, fillcolor="lightgray", penwidth=1];
    Topo [label="Geometric\\nFragmentation Filter"];
    Experts [label="10-Expert\\nClinical Panel"];
    RuleEngine [label="6-Rule\\nDecision Engine"];
    ToPart3 [label="To Decision Trees", shape=oval, fillcolor="lightgray", penwidth=1];
    CNNIn -> Topo -> Experts -> RuleEngine -> ToPart3;
}"""),

    ("Architecture_Part3.png", """digraph G3 {
    bgcolor="white"; rankdir=TB; nodesep=0.8; ranksep=0.6;
    node [shape=box, style="rounded,filled", fillcolor="white", color="black", penwidth=3, fontname="Helvetica-Bold", fontsize=28, margin="0.4,0.2"];
    edge [fontname="Helvetica-Bold", fontsize=24, color="black", penwidth=3];
    EngineIn [label="From Decision Engine", shape=oval, fillcolor="lightgray", penwidth=1];
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
    EngineIn -> DiffCheck;
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
    FinalReport -> UI -> End;
    { rank=same; DiffCheck; DiffOverride; }
    { rank=same; RPCheck; RPPrediction; }
    { rank=same; VetoCheck; Veto; }
}""")
]

for filename, dot_code in parts:
    encoded = urllib.parse.quote(dot_code)
    url = f"https://quickchart.io/graphviz?graph={encoded}&format=png"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    out_path = f"e:/V500/{filename}"
    try:
        with urllib.request.urlopen(req) as response, open(out_path, 'wb') as out_file:
            out_file.write(response.read())
        print(f"Downloaded {filename} successfully!")
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
