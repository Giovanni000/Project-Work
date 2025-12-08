#!/usr/bin/env python3
"""
Script to evaluate all images in the dataset and compare with ground truth labels.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch

# Import the classification function from the GUI
sys.path.insert(0, str(Path(__file__).parent))
from beko_detection_system import (
    OcclusionCNN, Teacher, Student, 
    preprocess_image, preprocess_image_for_efficientad,
    classify_connector, THRESHOLD_MULTIPLIERS
)

def load_models(models_dir, device):
    """Load all trained models."""
    print("Loading models...")
    
    # Load occlusion model
    occ_model = OcclusionCNN().to(device)
    occ_path = models_dir / "occlusion_cnn.pth"
    occ_model.load_state_dict(torch.load(occ_path, map_location=device))
    occ_model.eval()
    
    # Load EfficientAD-M models
    efficientad_models = {}
    connectors = [f"conn{i}" for i in range(1, 10)]
    
    for connector_name in connectors:
        student_path = models_dir / f"efficientad_student_{connector_name}.pth"
        threshold_path = models_dir / f"efficientad_threshold_{connector_name}.npy"
        
        if student_path.exists() and threshold_path.exists():
            teacher = Teacher().to(device)
            teacher.eval()
            
            student = Student().to(device)
            student.load_state_dict(torch.load(student_path, map_location=device))
            student.eval()
            
            threshold = np.load(threshold_path)
            
            # Apply multiplier
            multiplier = THRESHOLD_MULTIPLIERS.get(connector_name, 1.0)
            if multiplier != 1.0:
                threshold = threshold * multiplier
            
            efficientad_models[connector_name] = (teacher, student, threshold)
    
    print(f"✅ Loaded {len(efficientad_models)} EfficientAD-M models")
    return occ_model, efficientad_models

def evaluate_dataset(csv_path, models_dir, device=None):
    """Evaluate all images in the dataset."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Load dataset
    df = pd.read_csv(csv_path)
    print(f"\nLoaded {len(df)} images from dataset")
    
    # Load models
    occ_model, efficientad_models = load_models(models_dir, device)
    
    # Evaluate each image
    results = []
    
    # Fix paths: convert Google Drive paths to local paths
    project_root = Path(__file__).parent.parent
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        # Fix path: convert Google Drive path to local path
        original_path = str(row['image_path'])
        connector_name = row['connector_name']
        true_label = row['label']
        
        # Extract filename from path
        filename = Path(original_path).name
        # Use local path
        image_path = project_root / "Data" / "connectors" / connector_name / filename
        
        if not image_path.exists():
            # Skip if not found (silently)
            continue
        
        try:
            # Classify
            label, score, heatmap, _ = classify_connector(
                str(image_path),
                connector_name,
                occ_model,
                efficientad_models,
                device,
                use_efficientad=True
            )
            
            results.append({
                'image_path': str(image_path),
                'connector_name': connector_name,
                'true_label': true_label,
                'predicted_label': label,
                'score': score,
                'correct': true_label == label
            })
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
            results.append({
                'image_path': str(image_path),
                'connector_name': connector_name,
                'true_label': true_label,
                'predicted_label': 'ERROR',
                'score': 0.0,
                'correct': False
            })
    
    return pd.DataFrame(results)

def analyze_results(results_df):
    """Analyze and print results."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    # Overall accuracy
    total = len(results_df)
    correct = results_df['correct'].sum()
    accuracy = (correct / total) * 100 if total > 0 else 0
    
    print(f"\nOverall Accuracy: {correct}/{total} ({accuracy:.2f}%)")
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    print("-" * 40)
    confusion = pd.crosstab(
        results_df['true_label'], 
        results_df['predicted_label'],
        margins=True
    )
    print(confusion)
    
    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    print("-" * 40)
    for label in ['OK', 'KO', 'OCCLUSION']:
        label_df = results_df[results_df['true_label'] == label]
        if len(label_df) > 0:
            correct_count = label_df['correct'].sum()
            total_count = len(label_df)
            acc = (correct_count / total_count) * 100
            print(f"{label:12s}: {correct_count:4d}/{total_count:4d} ({acc:5.2f}%)")
    
    # Per-connector accuracy
    print("\nPer-Connector Accuracy:")
    print("-" * 40)
    connector_stats = []
    for conn in sorted(results_df['connector_name'].unique()):
        conn_df = results_df[results_df['connector_name'] == conn]
        correct_count = conn_df['correct'].sum()
        total_count = len(conn_df)
        acc = (correct_count / total_count) * 100 if total_count > 0 else 0
        
        # Count by true label
        ok_count = len(conn_df[conn_df['true_label'] == 'OK'])
        ko_count = len(conn_df[conn_df['true_label'] == 'KO'])
        occ_count = len(conn_df[conn_df['true_label'] == 'OCCLUSION'])
        
        connector_stats.append({
            'connector': conn,
            'accuracy': acc,
            'correct': correct_count,
            'total': total_count,
            'ok': ok_count,
            'ko': ko_count,
            'occlusion': occ_count
        })
        
        print(f"{conn:8s}: {correct_count:4d}/{total_count:4d} ({acc:5.2f}%) | OK:{ok_count:3d} KO:{ko_count:3d} OCC:{occ_count:3d}")
    
    # Error analysis
    print("\nError Analysis:")
    print("-" * 40)
    errors = results_df[~results_df['correct']]
    if len(errors) > 0:
        error_breakdown = pd.crosstab(
            errors['true_label'],
            errors['predicted_label']
        )
        print(error_breakdown)
    else:
        print("No errors!")
    
    return {
        'overall_accuracy': accuracy,
        'total': total,
        'correct': correct,
        'confusion_matrix': confusion,
        'connector_stats': connector_stats,
        'errors': errors
    }

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    csv_path = project_root / "Training" / "data" / "dataset.csv"
    models_dir = project_root / "Training" / "models"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Evaluate
    results_df = evaluate_dataset(csv_path, models_dir, device)
    
    # Analyze
    stats = analyze_results(results_df)
    
    # Save results
    output_path = project_root / "Training" / "evaluation_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n✅ Results saved to: {output_path}")
    
    # Save summary
    summary_path = project_root / "Training" / "evaluation_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("EVALUATION SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Overall Accuracy: {stats['overall_accuracy']:.2f}%\n")
        f.write(f"Total Images: {stats['total']}\n")
        f.write(f"Correct Predictions: {stats['correct']}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(stats['confusion_matrix']) + "\n\n")
        f.write("Per-Connector Statistics:\n")
        for stat in stats['connector_stats']:
            f.write(f"{stat['connector']}: {stat['accuracy']:.2f}% ({stat['correct']}/{stat['total']})\n")
    
    print(f"✅ Summary saved to: {summary_path}")

