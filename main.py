"""Entry point: runs the SHAP and counterfactual experiments.

Examples:
    python main.py --dataset LIVER
    python main.py --all
    python main.py --dataset GLASS --uncertainty entropy
"""
import argparse

from xai_uq import cf_xp, shap_xp

DATASET_LIST = [
    "BREAST_CANCER",
    "ECOLI",
    "GLASS",
    "HEART",
    "IONOSPHERE",
    "IRIS",
    "LIVER",
    "PARKINSON",
    "SONAR",
    "WINE",
]

# "eknn" is the method used in the paper; the others remain available.
UNCERTAINTY_CHOICES = ["eknn", "entropy", "density", "centroids"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Uncertainty vs. robustness of explanations (SHAP and counterfactuals)."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dataset",
        type=str.upper,
        choices=DATASET_LIST,
        help="Dataset name (case-insensitive).",
    )
    group.add_argument("--all", action="store_true", help="Run on all datasets.")
    parser.add_argument(
        "--uncertainty",
        nargs="+",
        default=["eknn"],
        choices=UNCERTAINTY_CHOICES,
        help="Uncertainty estimation method(s) (default: eknn).",
    )
    return parser.parse_args()


def run(dataset_name, uncertainties):
    for uncertainty in uncertainties:
        print(f"Running pipeline on dataset: {dataset_name} (uncertainty: {uncertainty})")
        shap_xp.robustness(uncertainty, dataset_name)
        cf_xp.robustness(uncertainty, dataset_name)


def main():
    args = parse_args()
    datasets = DATASET_LIST if args.all else [args.dataset]
    for dataset_name in datasets:
        run(dataset_name, args.uncertainty)


if __name__ == "__main__":
    main()
    print("Done!")
