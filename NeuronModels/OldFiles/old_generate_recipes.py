import json
from pathlib import Path

FEATURES = [
    "interburst_freq",
    "intraburst_freq",
    "duty_cycle",
    "mean_spikes_per_burst",
]
FITTING_DIR = Path(__file__).parent.parent / "hc_sweep/fitting"
RECIPE_DIR = FITTING_DIR / "recipes"
RECIPE_DIR.mkdir(exist_ok=True)

def generate_recipe_for_feature(feature: str):
    recommendations = []
    for param_dir in FITTING_DIR.iterdir():
        if not param_dir.is_dir():
            continue
        corr_path = param_dir / "correlation.json"
        if not corr_path.exists():
            continue
        with open(corr_path) as f:
            data = json.load(f)
        if feature in data:
            r = data[feature]
            abs_r = abs(r)
            if abs_r < 0.3:
                continue
            direction = "increase" if r > 0 else "decrease"
            strength = (
                "Strong" if abs_r > 0.7 else
                "Moderate" if abs_r > 0.4 else
                "Weak"
            )
            recommendations.append({
                "parameter": param_dir.name,
                "direction": direction,
                "justification": f"{strength} {'positive' if r > 0 else 'negative'} correlation (r = {r:.2f})"
            })
    out = {
        "feature": feature,
        "recommendation": recommendations
    }
    with open(RECIPE_DIR / f"{feature}.json", "w") as f:
        json.dump(out, f, indent=2)

def main():
    for feat in FEATURES:
        generate_recipe_for_feature(feat)

if __name__ == "__main__":
    main()
