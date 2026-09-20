import re

def parse_fn_patterns():
    fn_images = []
    
    with open("Evaluation_Results/V500_Evaluation_Report.txt", "r") as f:
        content = f.read()
        
    lines = content.split('\n')
    for line in lines:
        if " FN | " in line:
            # Example: [13/137] FN | Retinitis Pigmentosa11.jpg -> HEALTHY (AI: 29.9%)
            match = re.search(r'FN \| (Retinitis Pigmentosa\d+\.jpg) -> .*?\(AI: ([\d.]+)%\)', line)
            if match:
                img_name = match.group(1)
                ai_conf = float(match.group(2))
                fn_images.append((img_name, ai_conf))
                
    print(f"Found {len(fn_images)} False Negatives:")
    for img, ai in fn_images:
        print(f"  {img}: AI={ai}%")

if __name__ == "__main__":
    parse_fn_patterns()
