"""
ExplainRepair Experimental Evaluation Pipeline.
Executes Experiments A, B, C, D, and E with reproducible random seeds,
saving structured empirical results to results/ for dashboard visualization and analysis.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import torch

from config import default_config, RESULTS_DIR, DATA_DIR, CIFAR10_CLASSES
from explainers.gradcam_explainer import GradCAMExplainer
from explainers.integrated_gradients_explainer import IntegratedGradientsExplainer
from models.model import build_model, load_cifar10_dataloaders, train_model
from models.predictor import ModelPredictor
from reliability.reliability_evaluator import ReliabilityEvaluator
from repair.repair_engine import AdaptiveRepairEngine
from experiments.perturbation_tests import run_perturbation_stress_test


def parse_args():
    parser = argparse.ArgumentParser(description="Run ExplainRepair Research Experiments")
    parser.add_argument("--samples", type=int, default=15, help="Number of test samples to evaluate")
    parser.add_argument("--epochs", type=int, default=2, help="Epochs for quick model training if checkpoint absent")
    parser.add_argument("--quick", action="store_true", help="Quick mode for CPU testing (fewer steps)")
    parser.add_argument("--seed", type=int, default=42, help="Master random seed")
    return parser.parse_args()


def ensure_trained_model(args, device: torch.device) -> ModelPredictor:
    """Ensures model checkpoint exists; runs quick training if missing."""
    checkpoint_path = default_config.model.checkpoint_path
    predictor = ModelPredictor(checkpoint_path=str(checkpoint_path), device=device)

    if not predictor.model_status.get("is_trained", False):
        print("\n[ExplainRepair] No trained checkpoint found. Initializing quick training...")
        train_loader, val_loader = load_cifar10_dataloaders(
            data_dir=DATA_DIR,
            batch_size=64,
            train_subset_size=1000 if args.quick else 3000,
            val_subset_size=300 if args.quick else 600
        )
        history_path = RESULTS_DIR / "training_history.json"
        train_model(
            model=predictor.model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            learning_rate=0.001,
            device=device,
            checkpoint_path=checkpoint_path,
            history_path=history_path,
            progress_callback=lambda ep, tot, tl, ta, vl, va: print(
                f"  Epoch {ep}/{tot} | Train Loss: {tl:.4f} Acc: {ta*100:.1f}% | Val Loss: {vl:.4f} Acc: {va*100:.1f}%"
            )
        )
        # Reload predictor with saved checkpoint
        predictor = ModelPredictor(checkpoint_path=str(checkpoint_path), device=device)
        print(f"[ExplainRepair] Checkpoint saved successfully to {checkpoint_path}.\n")
    else:
        print(f"[ExplainRepair] Loaded existing trained model checkpoint from {checkpoint_path}.")

    return predictor


def run_pipeline(args):
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = default_config.device
    print(f"[ExplainRepair] Starting evaluation pipeline on device: {device}")
    
    predictor = ensure_trained_model(args, device)
    
    # Load evaluation samples
    _, val_loader = load_cifar10_dataloaders(
        data_dir=DATA_DIR,
        batch_size=1,
        val_subset_size=args.samples * 3  # Load enough to filter correct classifications
    )

    evaluator = ReliabilityEvaluator(default_config.reliability)
    repair_engine = AdaptiveRepairEngine(predictor=predictor, evaluator=evaluator, config=default_config)

    gradcam = GradCAMExplainer(predictor.model, target_layer_name="layer4", name="Grad-CAM (Layer 4)")
    ig = IntegratedGradientsExplainer(predictor.model, n_steps=30 if args.quick else 50, baseline_type="black", name="Integrated Gradients")

    raw_records: List[Dict[str, Any]] = []
    stress_test_records: List[Dict[str, Any]] = []
    
    correct_samples = 0
    total_evaluated = 0

    print(f"\n[ExplainRepair] Evaluating {args.samples} test samples...")

    start_exp_time = time.perf_counter()

    for sample_idx, (image_tensor, target_label) in enumerate(val_loader):
        if correct_samples >= args.samples:
            break

        image_tensor = image_tensor.to(device)
        true_class = int(target_label.item())

        pred_class, conf, _ = predictor.predict_tensor(image_tensor)
        # Filter for correctly classified samples for standard attribution evaluation
        if pred_class != true_class:
            continue

        correct_samples += 1
        total_evaluated += 1

        # Experiment A: Baseline Reliability
        t0 = time.perf_counter()
        gc_report = evaluator.evaluate(gradcam, image_tensor, predictor, target_class=pred_class, reference_explainer=ig, run_sanity_check=True)
        gc_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        ig_report = evaluator.evaluate(ig, image_tensor, predictor, target_class=pred_class, reference_explainer=gradcam, run_sanity_check=True)
        ig_time = time.perf_counter() - t0

        # Experiment B: Perturbation Stress Test (Track across all samples)
        s_res = run_perturbation_stress_test(
            gradcam, image_tensor, predictor,
            noise_levels=[0.02, 0.05, 0.10, 0.20, 0.35],
            seed=args.seed + correct_samples * 13
        )
        stress_test_records.append(s_res)

        # Experiment D: Adaptive Repair Loop with Decoupled Independent Validation
        t0 = time.perf_counter()
        repair_res = repair_engine.repair(
            initial_explainer=gradcam,
            tensor=image_tensor,
            target_class=pred_class,
            reference_explainer=ig
        )
        repair_time = time.perf_counter() - t0

        val_init_score = repair_res.independent_validation_initial.score if repair_res.independent_validation_initial else None
        val_sel_score = repair_res.independent_validation_selected.score if repair_res.independent_validation_selected else None

        raw_records.append({
            "sample_id": correct_samples,
            "true_class": CIFAR10_CLASSES[true_class],
            "pred_confidence": round(conf, 4),
            # Grad-CAM Metrics
            "gc_composite": gc_report.composite_score,
            "gc_stability": gc_report.stability.score,
            "gc_faithfulness": gc_report.faithfulness.score,
            "gc_agreement": gc_report.agreement.score if gc_report.agreement else None,
            "gc_sanity": gc_report.sanity_check.score if gc_report.sanity_check else None,
            "gc_time_sec": round(gc_time, 3),
            # IG Metrics
            "ig_composite": ig_report.composite_score,
            "ig_stability": ig_report.stability.score,
            "ig_faithfulness": ig_report.faithfulness.score,
            "ig_time_sec": round(ig_time, 3),
            # Repair Metrics
            "initial_method": repair_res.initial_explainer_name,
            "selected_method": repair_res.selected_explainer_name,
            "repair_outcome": repair_res.outcome,
            "initial_selection_score": repair_res.initial_report.composite_score,
            "repaired_selection_score": repair_res.selected_report.composite_score,
            "delta_selection": repair_res.delta_composite,
            "delta_stability": repair_res.delta_stability,
            "delta_faithfulness": repair_res.delta_faithfulness,
            "initial_validation_score": val_init_score,
            "repaired_validation_score": val_sel_score,
            "delta_validation": repair_res.delta_validation,
            "validation_outcome": repair_res.validation_outcome,
            "repair_time_sec": round(repair_time, 3),
            "diagnoses_count": len(repair_res.diagnosed_failures),
            "diagnoses": "; ".join(repair_res.diagnosed_failures)
        })

        print(f"  Sample {correct_samples}/{args.samples} ({CIFAR10_CLASSES[true_class]}): "
              f"GC={gc_report.composite_score:.2f} | IG={ig_report.composite_score:.2f} | "
              f"Repair Outcome={repair_res.outcome} (Δ Sel: {repair_res.delta_composite:+.2f}, "
              f"Val: {repair_res.validation_outcome})")

    total_duration = time.perf_counter() - start_exp_time

    # Compile DataFrame
    df = pd.DataFrame(raw_records)
    csv_path = RESULTS_DIR / "experiments_raw.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[ExplainRepair] Raw experiment observations saved to {csv_path}")

    # Aggregate summaries with explicit denominator distinctions
    outcomes = df["repair_outcome"].value_counts().to_dict()
    val_outcomes = df["validation_outcome"].value_counts().to_dict()

    total_samples = len(df)
    no_repair_needed_count = int(outcomes.get("NO_REPAIR_NEEDED", 0))
    samples_requiring_repair = total_samples - no_repair_needed_count
    actual_repair_attempts = samples_requiring_repair

    success_count = int(outcomes.get("SUCCESS", 0))
    partial_count = int(outcomes.get("PARTIAL_IMPROVEMENT", 0))
    unsuccessful_count = int(outcomes.get("UNSUCCESSFUL", 0))
    validated_success_count = int(val_outcomes.get("VALIDATED_IMPROVEMENT", 0))

    # Correct rates with explicit stated denominators
    intervention_success_rate = round(float(success_count / actual_repair_attempts * 100), 1) if actual_repair_attempts > 0 else 0.0
    population_repair_rate = round(float(success_count / total_samples * 100), 1) if total_samples > 0 else 0.0
    independent_validation_rate = round(float(validated_success_count / actual_repair_attempts * 100), 1) if actual_repair_attempts > 0 else 0.0

    # Aggregate Experiment B stress test curves across all samples
    stress_summary = {}
    if stress_test_records:
        noise_lvls = stress_test_records[0]["noise_levels"]
        avg_cosines = []
        avg_ranks = []
        for i_lvl in range(len(noise_lvls)):
            cos_vals = [rec["cosine_similarities"][i_lvl] for rec in stress_test_records]
            rank_vals = [rec["rank_correlations"][i_lvl] for rec in stress_test_records]
            avg_cosines.append(round(float(np.mean(cos_vals)), 4))
            avg_ranks.append(round(float(np.mean(rank_vals)), 4))
        stress_summary = {
            "noise_levels": noise_lvls,
            "mean_cosine_similarities": avg_cosines,
            "mean_rank_correlations": avg_ranks,
            "samples_averaged": len(stress_test_records)
        }

    # Statistical standard error calculation
    gc_sem = round(float(df["gc_composite"].std() / np.sqrt(total_samples)), 4) if total_samples > 1 else 0.0
    ig_sem = round(float(df["ig_composite"].std() / np.sqrt(total_samples)), 4) if total_samples > 1 else 0.0

    summary = {
        "metadata": {
            "num_samples": total_samples,
            "device": str(device),
            "duration_seconds": round(total_duration, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "baseline_comparison": {
            "gradcam": {
                "composite_mean": round(float(df["gc_composite"].mean()), 4),
                "composite_std": round(float(df["gc_composite"].std()), 4),
                "composite_sem": gc_sem,
                "stability_mean": round(float(df["gc_stability"].mean()), 4),
                "faithfulness_mean": round(float(df["gc_faithfulness"].mean()), 4),
                "avg_runtime_sec": round(float(df["gc_time_sec"].mean()), 3)
            },
            "integrated_gradients": {
                "composite_mean": round(float(df["ig_composite"].mean()), 4),
                "composite_std": round(float(df["ig_composite"].std()), 4),
                "composite_sem": ig_sem,
                "stability_mean": round(float(df["ig_stability"].mean()), 4),
                "faithfulness_mean": round(float(df["ig_faithfulness"].mean()), 4),
                "avg_runtime_sec": round(float(df["ig_time_sec"].mean()), 3)
            }
        },
        "repair_cohort_breakdown": {
            "total_samples_evaluated": total_samples,
            "samples_requiring_repair": samples_requiring_repair,
            "samples_requiring_no_repair": no_repair_needed_count,
            "actual_repair_attempts": actual_repair_attempts,
            "selection_improved_attempts": success_count,
            "partial_improvement_attempts": partial_count,
            "unsuccessful_repair_attempts": unsuccessful_count,
            "independent_validated_attempts": validated_success_count,
            "intervention_success_rate_pct": intervention_success_rate,
            "intervention_success_rate_denominator": "actual_repair_attempts",
            "independent_validation_rate_pct": independent_validation_rate,
            "independent_validation_rate_denominator": "actual_repair_attempts",
            "population_repair_rate_pct": population_repair_rate,
            "population_repair_rate_denominator": "total_samples_evaluated",
            "mean_selection_improvement_delta": round(float(df["delta_selection"].mean()), 4),
            "mean_validation_improvement_delta": round(float(df["delta_validation"].dropna().mean()), 4) if "delta_validation" in df else 0.0,
            "avg_repair_runtime_sec": round(float(df["repair_time_sec"].mean()), 3)
        },
        "stress_test_spectrum": stress_summary
    }

    json_path = RESULTS_DIR / "experiments_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[ExplainRepair] Summary metrics saved to {json_path}")

    cohort = summary["repair_cohort_breakdown"]
    print("\n================== EXPERIMENTAL RESULTS SUMMARY ==================")
    print(f"Evaluated Samples: {total_samples} CIFAR-10 images")
    print(f"Grad-CAM Mean Reliability:      {summary['baseline_comparison']['gradcam']['composite_mean']:.3f} ± {gc_sem:.3f} (SEM)")
    print(f"Integrated Gradients Mean Rel.:  {summary['baseline_comparison']['integrated_gradients']['composite_mean']:.3f} ± {ig_sem:.3f} (SEM)")
    print(f"\nCohort & Repair Outcome Accounting:")
    print(f"  • Total Evaluated Samples:      {cohort['total_samples_evaluated']}")
    print(f"  • Samples Requiring Repair:     {cohort['samples_requiring_repair']}")
    print(f"  • Healthy (No Repair Needed):   {cohort['samples_requiring_no_repair']}")
    print(f"  • Actual Repair Attempts:       {cohort['actual_repair_attempts']}")
    print(f"  • Selection Score Improved (Δ): {cohort['selection_improved_attempts']} ({cohort['intervention_success_rate_pct']}% of attempts)")
    print(f"  • Independent Validation Passed:{cohort['independent_validated_attempts']} ({cohort['independent_validation_rate_pct']}% of attempts)")
    print(f"  • Unsuccessful Attempts:        {cohort['unsuccessful_repair_attempts']}")
    print(f"Mean Selection Score Gain (Δ):    +{cohort['mean_selection_improvement_delta']:.3f}")
    print(f"Mean Validation Score Gain (Δ):   +{cohort['mean_validation_improvement_delta']:.3f}")
    print(f"Avg Repair Runtime Overhead:      {cohort['avg_repair_runtime_sec']:.2f} s/image")
    print("==================================================================\n")


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
