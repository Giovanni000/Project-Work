#!/usr/bin/env python3
"""
Script per testare i modelli EfficientAD-M addestrati in models/Training n 2/
Testa solo su casi OK e KO (esclude OCCLUSION).
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import json

# Import threshold multipliers from beko_detection_system
sys.path.insert(0, str(Path(__file__).parent))
from beko_detection_system import THRESHOLD_MULTIPLIERS

# Define Teacher and Student classes with multi-layer architecture (matching Training n 2)
import ssl
from urllib.error import URLError
from torchvision.models import resnet18, ResNet18_Weights

class Teacher(nn.Module):
    """Teacher model: ResNet18 pre-trained, frozen, multi-layer."""
    def __init__(self):
        super(Teacher, self).__init__()
        try:
            base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except (ssl.SSLError, URLError) as e:
            print(f"⚠️  SSL/URL Error loading ResNet18 pre-trained weights: {e}")
            print("   Attempting to load with disabled SSL verification...")
            try:
                ssl._create_default_https_context = ssl._create_unverified_context
                base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
                print("   ✅ ResNet18 pre-trained weights loaded with SSL verification disabled.")
            except Exception as inner_e:
                print(f"❌ Failed to load ResNet18 pre-trained weights even with SSL disabled: {inner_e}")
                print("   Using ResNet18 without pre-trained weights (performance might be reduced).")
                base = resnet18(weights=None)
        except Exception as e:
            print(f"❌ Unexpected error loading ResNet18 pre-trained weights: {e}")
            print("   Using ResNet18 without pre-trained weights (performance might be reduced).")
            base = resnet18(weights=None)
        
        # Decompose into layers
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        x = self.stem(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return {
            "layer2": x2,
            "layer3": x3,
            "layer4": x4,
        }


class Student(nn.Module):
    """Student model: ResNet18 non-pre-trained, trainable, multi-layer."""
    def __init__(self):
        super(Student, self).__init__()
        base = resnet18(weights=None)  # No pre-training
        
        # Decompose into layers
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
    
    def forward(self, x):
        x = self.stem(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return {
            "layer2": x2,
            "layer3": x3,
            "layer4": x4,
        }


def preprocess_image_for_efficientad(image_path, device):
    """Preprocess image for EfficientAD-M (ImageNet normalization)."""
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(device)
    return x

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

MODELS_DIR = Path(__file__).parent / "models" / "Training n 2"
CSV_PATH = Path(__file__).parent / "data" / "dataset.csv"
OUTPUT_DIR = Path(__file__).parent / "test_results_training2"
OUTPUT_DIR.mkdir(exist_ok=True)

# Configurazione EfficientAD-M (deve corrispondere al training)
FEATURE_LAYERS = ["layer2", "layer3", "layer4"]
FEATURE_LAYER_WEIGHTS = {
    "layer2": 1.0,
    "layer3": 1.0,
    "layer4": 1.0,
}
TOP_K_PERCENT = 0.01
IMG_SIZE = 128

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_spatial_weight_mask(connector_name, target_size, save_dir, device):
    """Load spatial mask (se esiste)."""
    import torch.nn.functional as F
    
    models_dir = Path(save_dir)
    mask_path = models_dir / f"spatial_mask_{connector_name}.npy"
    
    if not mask_path.exists():
        return torch.ones(1, 1, target_size[0], target_size[1], device=device)
    
    mask = np.load(mask_path)
    mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
    
    if mask_tensor.shape[2:] != target_size:
        mask_tensor = F.interpolate(
            mask_tensor, 
            size=target_size, 
            mode='bilinear', 
            align_corners=False
        )
    
    return mask_tensor.to(device)


def compute_fused_anomaly_map(teacher_feats, student_feats,
                              spatial_mask_fullres,
                              feature_layers=FEATURE_LAYERS,
                              feature_layer_weights=FEATURE_LAYER_WEIGHTS,
                              device=None):
    """Compute fused 2D anomaly map."""
    import torch.nn.functional as F
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ref_layer = feature_layers[0]
    ref_feat = teacher_feats[ref_layer]
    _, _, Href, Wref = ref_feat.shape
    
    fused = None
    
    for layer_name in feature_layers:
        t = teacher_feats[layer_name]
        s = student_feats[layer_name]
        
        diff = (t - s) ** 2
        amap = diff.mean(dim=1, keepdim=True)
        
        _, _, Hf, Wf = amap.shape
        
        mask_resized = F.interpolate(
            spatial_mask_fullres,
            size=(Hf, Wf),
            mode='bilinear',
            align_corners=False
        )
        
        amap = amap * mask_resized
        
        if (Hf, Wf) != (Href, Wref):
            amap = F.interpolate(
                amap,
                size=(Href, Wref),
                mode='bilinear',
                align_corners=False
            )
        
        layer_weight = feature_layer_weights.get(layer_name, 1.0)
        if fused is None:
            fused = layer_weight * amap
        else:
            fused = fused + layer_weight * amap
    
    return fused


def compute_anomaly_score_from_features(teacher_feats, student_feats,
                                        spatial_mask_fullres,
                                        feature_layers=FEATURE_LAYERS,
                                        feature_layer_weights=FEATURE_LAYER_WEIGHTS,
                                        topk_percent=TOP_K_PERCENT,
                                        device=None):
    """Compute robust anomaly score."""
    fused = compute_fused_anomaly_map(
        teacher_feats, student_feats,
        spatial_mask_fullres,
        feature_layers=feature_layers,
        feature_layer_weights=feature_layer_weights,
        device=device
    )
    
    B, _, Href, Wref = fused.shape
    fused_flat = fused.view(B, -1)
    
    k = max(1, int(topk_percent * fused_flat.size(1)))
    topk_vals, _ = torch.topk(fused_flat, k=k, dim=1)
    scores = topk_vals.mean(dim=1)
    
    return scores


def load_efficientad_model(connector_name, models_dir, device):
    """Load EfficientAD-M model from Training n 2."""
    model_path = models_dir / f"efficientad_student_{connector_name}.pth"
    threshold_path = models_dir / f"efficientad_threshold_{connector_name}.npy"
    spatial_mask_path = models_dir / f"spatial_mask_{connector_name}.npy"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Modello non trovato: {model_path}")
    
    if not threshold_path.exists():
        raise FileNotFoundError(f"Threshold non trovato: {threshold_path}")
    
    # Load Teacher (multi-layer)
    teacher = Teacher().to(device)
    teacher.eval()
    
    # Load Student (multi-layer)
    student = Student().to(device)
    student.load_state_dict(torch.load(model_path, map_location=device))
    student.eval()
    
    # Load threshold
    threshold = np.load(threshold_path)
    
    # Apply multiplier
    multiplier = THRESHOLD_MULTIPLIERS.get(connector_name, 1.0)
    if multiplier != 1.0:
        threshold = threshold * multiplier
    
    # Check spatial mask
    spatial_mask_exists = spatial_mask_path.exists()
    
    return teacher, student, threshold, spatial_mask_exists


def classify_image(image_path, connector_name, teacher, student, threshold, 
                   spatial_mask_exists, device):
    """Classify a single image."""
    # Load spatial mask
    try:
        spatial_mask_fullres = load_spatial_weight_mask(
            connector_name,
            target_size=(IMG_SIZE, IMG_SIZE),
            save_dir=MODELS_DIR,
            device=device
        )
    except:
        spatial_mask_fullres = torch.ones(1, 1, IMG_SIZE, IMG_SIZE, device=device)
    
    # Preprocess
    x = preprocess_image_for_efficientad(image_path, device)
    
    # Get features
    with torch.no_grad():
        teacher_feats = teacher(x)
        student_feats = student(x)
        
        # Compute score
        scores = compute_anomaly_score_from_features(
            teacher_feats,
            student_feats,
            spatial_mask_fullres,
            feature_layers=FEATURE_LAYERS,
            feature_layer_weights=FEATURE_LAYER_WEIGHTS,
            topk_percent=TOP_K_PERCENT,
            device=device
        )
        
        score = scores[0].cpu().item()
    
    # Decision
    label = "KO" if score > threshold else "OK"
    
    return label, score, threshold


# ============================================================================
# MAIN TEST FUNCTION
# ============================================================================

def test_models():
    """Test all models on OK and KO images only."""
    
    print("="*70)
    print("TEST MODELLI EFFICIENTAD-M - Training n 2")
    print("="*70)
    print(f"\nModelli directory: {MODELS_DIR}")
    print(f"CSV dataset: {CSV_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Check models directory
    if not MODELS_DIR.exists():
        raise FileNotFoundError(f"Directory modelli non trovata: {MODELS_DIR}")
    
    # Load dataset
    print(f"\n📊 Caricamento dataset...")
    df = pd.read_csv(CSV_PATH)
    print(f"   Totale immagini: {len(df)}")
    
    # Filter: only OK and KO (exclude OCCLUSION)
    df_filtered = df[df['label'].isin(['OK', 'KO'])].copy()
    print(f"   Immagini OK+KO (senza OCCLUSION): {len(df_filtered)}")
    print(f"   Distribuzione:")
    print(df_filtered['label'].value_counts())
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")
    
    # Load models for each connector
    connectors = [f"conn{i}" for i in range(1, 10)]
    models = {}
    
    print(f"\n📦 Caricamento modelli...")
    for connector_name in connectors:
        try:
            teacher, student, threshold, has_mask = load_efficientad_model(
                connector_name, MODELS_DIR, device
            )
            models[connector_name] = (teacher, student, threshold, has_mask)
            print(f"   ✅ {connector_name}: threshold={threshold:.6f}, mask={'✅' if has_mask else '❌'}")
        except Exception as e:
            print(f"   ❌ {connector_name}: {e}")
    
    if len(models) == 0:
        raise ValueError("Nessun modello caricato!")
    
    # Test each connector
    print(f"\n🔍 Testing modelli...")
    all_results = []
    
    for connector_name in connectors:
        if connector_name not in models:
            continue
        
        print(f"\n{'='*70}")
        print(f"Testing {connector_name}")
        print(f"{'='*70}")
        
        teacher, student, threshold, has_mask = models[connector_name]
        
        # Filter connector images
        conn_df = df_filtered[df_filtered['connector_name'] == connector_name].copy()
        
        if len(conn_df) == 0:
            print(f"   ⚠️  Nessuna immagine OK/KO per {connector_name}")
            continue
        
        print(f"   Immagini da testare: {len(conn_df)}")
        print(f"   Distribuzione: {conn_df['label'].value_counts().to_dict()}")
        
        # Test images
        results = []
        for idx, row in tqdm(conn_df.iterrows(), total=len(conn_df), desc=f"  Testing {connector_name}"):
            image_path = Path(row['image_path'])
            true_label = row['label']
            
            # Fix path if needed
            if not image_path.exists():
                # Try local path
                local_path = Path(__file__).parent.parent / "Data" / "connectors" / connector_name / image_path.name
                if local_path.exists():
                    image_path = local_path
                else:
                    print(f"     ⚠️  Immagine non trovata: {image_path}")
                    continue
            
            try:
                pred_label, score, thresh = classify_image(
                    str(image_path),
                    connector_name,
                    teacher,
                    student,
                    threshold,
                    has_mask,
                    device
                )
                
                results.append({
                    'connector': connector_name,
                    'image_path': str(image_path),
                    'true_label': true_label,
                    'predicted_label': pred_label,
                    'score': score,
                    'threshold': thresh,
                    'correct': true_label == pred_label
                })
            except Exception as e:
                print(f"     ❌ Errore su {image_path.name}: {e}")
                continue
        
        all_results.extend(results)
        
        # Per-connector statistics
        if results:
            results_df = pd.DataFrame(results)
            correct = results_df['correct'].sum()
            total = len(results_df)
            accuracy = (correct / total) * 100
            
            print(f"\n   📊 Risultati {connector_name}:")
            print(f"      Accuracy: {correct}/{total} ({accuracy:.2f}%)")
            
            # Confusion matrix
            cm = pd.crosstab(results_df['true_label'], results_df['predicted_label'])
            print(f"      Confusion Matrix:")
            print(f"      {cm}")
    
    # Overall analysis
    print(f"\n{'='*70}")
    print("ANALISI COMPLESSIVA")
    print(f"{'='*70}")
    
    if len(all_results) == 0:
        print("❌ Nessun risultato da analizzare!")
        return
    
    results_df = pd.DataFrame(all_results)
    
    # Overall accuracy
    total = len(results_df)
    correct = results_df['correct'].sum()
    accuracy = (correct / total) * 100
    
    print(f"\n📊 Statistiche Generali:")
    print(f"   Totale immagini testate: {total}")
    print(f"   Corrette: {correct}")
    print(f"   Accuracy complessiva: {accuracy:.2f}%")
    
    # Confusion matrix
    print(f"\n📋 Confusion Matrix:")
    cm = pd.crosstab(results_df['true_label'], results_df['predicted_label'], margins=True)
    print(cm)
    
    # Per-class accuracy
    print(f"\n📊 Accuracy per Classe:")
    for label in ['OK', 'KO']:
        label_df = results_df[results_df['true_label'] == label]
        if len(label_df) > 0:
            correct_count = label_df['correct'].sum()
            total_count = len(label_df)
            acc = (correct_count / total_count) * 100
            print(f"   {label:12s}: {correct_count:4d}/{total_count:4d} ({acc:5.2f}%)")
    
    # Per-connector accuracy
    print(f"\n📊 Accuracy per Connettore:")
    connector_stats = []
    for conn in sorted(results_df['connector'].unique()):
        conn_df = results_df[results_df['connector'] == conn]
        correct_count = conn_df['correct'].sum()
        total_count = len(conn_df)
        acc = (correct_count / total_count) * 100 if total_count > 0 else 0
        
        ok_count = len(conn_df[conn_df['true_label'] == 'OK'])
        ko_count = len(conn_df[conn_df['true_label'] == 'KO'])
        
        connector_stats.append({
            'connector': conn,
            'accuracy': acc,
            'correct': correct_count,
            'total': total_count,
            'ok': ok_count,
            'ko': ko_count
        })
        
        print(f"   {conn:8s}: {correct_count:4d}/{total_count:4d} ({acc:5.2f}%) | OK:{ok_count:3d} KO:{ko_count:3d}")
    
    # Error analysis
    print(f"\n📊 Analisi Errori:")
    errors = results_df[~results_df['correct']]
    if len(errors) > 0:
        error_breakdown = pd.crosstab(errors['true_label'], errors['predicted_label'])
        print(error_breakdown)
        
        print(f"\n   Errori per tipo:")
        ok_to_ko = len(errors[(errors['true_label'] == 'OK') & (errors['predicted_label'] == 'KO')])
        ko_to_ok = len(errors[(errors['true_label'] == 'KO') & (errors['predicted_label'] == 'OK')])
        print(f"      OK → KO (False Positive): {ok_to_ko}")
        print(f"      KO → OK (False Negative): {ko_to_ok}")
    else:
        print("   ✅ Nessun errore!")
    
    # Score analysis
    print(f"\n📊 Analisi Score:")
    ok_scores = results_df[results_df['true_label'] == 'OK']['score'].values
    ko_scores = results_df[results_df['true_label'] == 'KO']['score'].values
    
    if len(ok_scores) > 0:
        print(f"   OK scores: mean={np.mean(ok_scores):.6f}, std={np.std(ok_scores):.6f}")
        print(f"              min={np.min(ok_scores):.6f}, max={np.max(ok_scores):.6f}")
    
    if len(ko_scores) > 0:
        print(f"   KO scores: mean={np.mean(ko_scores):.6f}, std={np.std(ko_scores):.6f}")
        print(f"              min={np.min(ko_scores):.6f}, max={np.max(ko_scores):.6f}")
    
    # Save results
    print(f"\n💾 Salvataggio risultati...")
    
    # Save CSV
    results_df.to_csv(OUTPUT_DIR / "test_results.csv", index=False)
    print(f"   ✅ Risultati salvati: {OUTPUT_DIR / 'test_results.csv'}")
    
    # Save summary
    summary = {
        'overall_accuracy': accuracy,
        'total_images': total,
        'correct_predictions': correct,
        'confusion_matrix': cm.to_dict(),
        'per_class_accuracy': {
            label: {
                'correct': len(results_df[(results_df['true_label'] == label) & results_df['correct']]),
                'total': len(results_df[results_df['true_label'] == label]),
                'accuracy': len(results_df[(results_df['true_label'] == label) & results_df['correct']]) / len(results_df[results_df['true_label'] == label]) * 100 if len(results_df[results_df['true_label'] == label]) > 0 else 0
            }
            for label in ['OK', 'KO']
        },
        'connector_stats': connector_stats,
        'error_analysis': {
            'ok_to_ko': len(errors[(errors['true_label'] == 'OK') & (errors['predicted_label'] == 'KO')]) if len(errors) > 0 else 0,
            'ko_to_ok': len(errors[(errors['true_label'] == 'KO') & (errors['predicted_label'] == 'OK')]) if len(errors) > 0 else 0
        },
        'score_statistics': {
            'ok': {
                'mean': float(np.mean(ok_scores)) if len(ok_scores) > 0 else 0,
                'std': float(np.std(ok_scores)) if len(ok_scores) > 0 else 0,
                'min': float(np.min(ok_scores)) if len(ok_scores) > 0 else 0,
                'max': float(np.max(ok_scores)) if len(ok_scores) > 0 else 0
            },
            'ko': {
                'mean': float(np.mean(ko_scores)) if len(ko_scores) > 0 else 0,
                'std': float(np.std(ko_scores)) if len(ko_scores) > 0 else 0,
                'min': float(np.min(ko_scores)) if len(ko_scores) > 0 else 0,
                'max': float(np.max(ko_scores)) if len(ko_scores) > 0 else 0
            }
        }
    }
    
    with open(OUTPUT_DIR / "test_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"   ✅ Summary salvato: {OUTPUT_DIR / 'test_summary.json'}")
    
    # Visualizations
    print(f"\n📈 Creazione visualizzazioni...")
    
    # 1. Confusion Matrix Heatmap
    plt.figure(figsize=(10, 8))
    cm_plot = pd.crosstab(results_df['true_label'], results_df['predicted_label'])
    sns.heatmap(cm_plot, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - OK/KO Classification\n(Excluding OCCLUSION)', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Confusion matrix: {OUTPUT_DIR / 'confusion_matrix.png'}")
    
    # 2. Per-Connector Accuracy Bar Plot
    plt.figure(figsize=(12, 6))
    conn_acc = pd.DataFrame(connector_stats)
    plt.bar(conn_acc['connector'], conn_acc['accuracy'], color='steelblue', edgecolor='black')
    plt.axhline(y=accuracy, color='r', linestyle='--', label=f'Overall: {accuracy:.2f}%')
    plt.xlabel('Connector', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Per-Connector Accuracy', fontsize=14, fontweight='bold')
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "per_connector_accuracy.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Per-connector accuracy: {OUTPUT_DIR / 'per_connector_accuracy.png'}")
    
    # 3. Score Distribution
    plt.figure(figsize=(12, 6))
    plt.hist(ok_scores, bins=50, alpha=0.7, label='OK', color='green', edgecolor='black')
    plt.hist(ko_scores, bins=50, alpha=0.7, label='KO', color='red', edgecolor='black')
    plt.xlabel('Anomaly Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Anomaly Score Distribution: OK vs KO', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "score_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Score distribution: {OUTPUT_DIR / 'score_distribution.png'}")
    
    # 4. Per-Connector Score Comparison
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    for idx, conn in enumerate(sorted(results_df['connector'].unique())):
        conn_df = results_df[results_df['connector'] == conn]
        conn_ok = conn_df[conn_df['true_label'] == 'OK']['score'].values
        conn_ko = conn_df[conn_df['true_label'] == 'KO']['score'].values
        
        axes[idx].hist(conn_ok, bins=30, alpha=0.7, label='OK', color='green', edgecolor='black')
        axes[idx].hist(conn_ko, bins=30, alpha=0.7, label='KO', color='red', edgecolor='black')
        axes[idx].set_title(f'{conn}', fontweight='bold')
        axes[idx].set_xlabel('Score')
        axes[idx].set_ylabel('Frequency')
        axes[idx].legend()
        axes[idx].grid(alpha=0.3)
    
    plt.suptitle('Anomaly Score Distribution per Connettore', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "score_distribution_per_connector.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Score distribution per connector: {OUTPUT_DIR / 'score_distribution_per_connector.png'}")
    
    print(f"\n✅ Test completato!")
    print(f"   Risultati salvati in: {OUTPUT_DIR}")
    
    return results_df, summary


if __name__ == "__main__":
    results_df, summary = test_models()

