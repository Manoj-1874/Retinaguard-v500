with open(r"e:\V500\app.py", "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

def print_func(func_name):
    for i, line in enumerate(lines, 1):
        if f"def {func_name}" in line:
            print(f"--- {func_name} ---")
            for idx in range(i - 1, min(len(lines), i + 40)):
                print(f"  {idx+1}: {lines[idx].strip()}")
            break

print_func("vessel_attenuation_expert")
print_func("pigment_bone_spicules_expert")
print_func("spatial_pattern_expert")
