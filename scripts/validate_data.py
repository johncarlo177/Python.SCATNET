import os, sys

datasets = {
    "indian_pines": ["Indian_pines_corrected.mat", "Indian_pines_gt.mat"],
    "pavia_centre": ["Pavia.mat", "Pavia_gt.mat"],
    # "houston2013": ["Houston.mat", "Houston_gt.mat"]
}

base = os.path.join(os.path.dirname(__file__), "..", "data")
print("Checking datasets in:", os.path.abspath(base))

missing = []
for name, files in datasets.items():
    folder = os.path.join(base, name)
    for f in files:
        path = os.path.join(folder, f)
        if not os.path.exists(path):
            missing.append(path)

if missing:
    print("\n⚠️ WARNING: Some dataset files are missing:")
    for m in missing:
        print(" -", m)
    print("\nContinuing execution anyway (using placeholder training + synthetic outputs)...\n")
else:
    print("\n✅ All required dataset files are present.")


    