#!/usr/bin/env python3
"""
BEKO DETECTION SYSTEM
Interfaccia grafica per analisi PCB connector con modelli deep learning.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image, ImageTk
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage
from pathlib import Path
from dataclasses import dataclass
import tempfile
import threading
import sys

# ============================================================================
# CLASSI MODELLI (da step1 e step2)
# ============================================================================

class OcclusionCNN(nn.Module):
    """CNN per classificazione OCCLUSION vs VISIBLE."""
    def __init__(self):
        super(OcclusionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class ConvAE(nn.Module):
    """Autoencoder per anomaly detection (LEGACY - non più usato)."""
    def __init__(self):
        super(ConvAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# ============================================================================
# MODELLI EFFICIENTAD-M (nuovo sistema)
# ============================================================================

class Teacher(nn.Module):
    """
    Teacher: ResNet18 pre-addestrato su ImageNet (congelato).
    Usato per estrarre feature di riferimento.
    """
    def __init__(self):
        super(Teacher, self).__init__()
        # Carica ResNet18 pre-addestrato con gestione errori SSL
        try:
            # Prova a caricare con pesi pre-addestrati
            base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except Exception as e:
            # Se fallisce (es. errore SSL), prova senza verifica SSL
            import ssl
            import urllib.request
            print(f"⚠️  Errore caricamento pesi pre-addestrati: {e}")
            print("   Tentativo con verifica SSL disabilitata...")
            
            # Disabilita temporaneamente verifica SSL per il download
            original_ssl_context = ssl._create_default_https_context
            ssl._create_default_https_context = ssl._create_unverified_context
            
            try:
                base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
                print("   ✅ Pesi caricati con successo")
            except Exception as e2:
                print(f"   ❌ Fallito anche senza verifica SSL: {e2}")
                print("   ⚠️  Uso ResNet18 senza pesi pre-addestrati (prestazioni ridotte)")
                base = resnet18(weights=None)
            finally:
                # Ripristina contesto SSL originale
                ssl._create_default_https_context = original_ssl_context
        
        # Rimuovi layer finali (avgpool e fc), mantieni solo encoder
        self.encoder = nn.Sequential(*list(base.children())[:-2])
        
        # Congela tutti i parametri (no training)
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        """
        Args:
            x: Tensor [B, 3, H, W]
        Returns:
            feature_map: Tensor [B, 512, Hf, Wf] dove Hf=H/32, Wf=W/32
        """
        return self.encoder(x)


class Student(nn.Module):
    """
    Student: ResNet18 NON pre-addestrato (pesi random).
    Addestrato per imitare le feature del Teacher su immagini OK.
    """
    def __init__(self):
        super(Student, self).__init__()
        # Carica ResNet18 SENZA pesi pre-addestrati
        base = resnet18(weights=None)
        # Rimuovi layer finali (avgpool e fc), mantieni solo encoder
        self.encoder = nn.Sequential(*list(base.children())[:-2])
    
    def forward(self, x):
        """
        Args:
            x: Tensor [B, 3, H, W]
        Returns:
            feature_map: Tensor [B, 512, Hf, Wf] dove Hf=H/32, Wf=W/32
        """
        return self.encoder(x)

# ============================================================================
# FUNZIONI PREPROCESSING
# ============================================================================

@dataclass
class RelativeROI:
    name: str
    x_min_rel: float
    y_min_rel: float
    x_max_rel: float
    y_max_rel: float
    
    def to_pixel_box(self, width: int, height: int, margin: int = 0):
        x_min = int(self.x_min_rel * width)
        y_min = int(self.y_min_rel * height)
        x_max = int(self.x_max_rel * width)
        y_max = int(self.y_max_rel * height)
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(width, x_max + margin)
        y_max = min(height, y_max + margin)
        return x_min, y_min, x_max, y_max

def load_roi_config(roi_config_path):
    """Carica configurazione ROI da JSON."""
    with open(roi_config_path, 'r') as f:
        data = json.load(f)
    rois = []
    for entry in data:
        rois.append(RelativeROI(
            name=entry["name"],
            x_min_rel=float(entry["x_min_rel"]),
            y_min_rel=float(entry["y_min_rel"]),
            x_max_rel=float(entry["x_max_rel"]),
            y_max_rel=float(entry["y_max_rel"])
        ))
    return rois

def detect_homography(template_gray, candidate_gray, max_features=4000, good_match_percent=0.15):
    """Rileva omografia tra template e candidato."""
    orb = cv2.ORB_create(nfeatures=max_features, fastThreshold=5, scaleFactor=1.2)
    kp1, des1 = orb.detectAndCompute(template_gray, None)
    kp2, des2 = orb.detectAndCompute(candidate_gray, None)
    
    if des1 is None or des2 is None:
        raise RuntimeError("Could not extract ORB descriptors")
    
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    
    if len(matches) < 15:
        knn = matcher.knnMatch(des1, des2, k=2)
        matches = []
        for m, n in knn:
            if m.distance < 0.75 * n.distance:
                matches.append(m)
    
    if not matches:
        raise RuntimeError("No matches found")
    
    matches = sorted(matches, key=lambda m: m.distance)
    keep = max(4, int(len(matches) * good_match_percent))
    matches = matches[:keep]
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 4.0)
    if H is None or mask is None or mask.sum() < 8:
        raise RuntimeError("Homography estimation failed")
    return H

def normalize_lighting(image_bgr):
    """Normalizza l'illuminazione dell'immagine."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    balanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Gray-world white balance
    avg_b = np.mean(balanced[:, :, 0])
    avg_g = np.mean(balanced[:, :, 1])
    avg_r = np.mean(balanced[:, :, 2])
    avg_gray = (avg_b + avg_g + avg_r) / 3.0
    scale = np.array([avg_gray / avg_b, avg_gray / avg_g, avg_gray / avg_r])
    wb = balanced.astype(np.float32)
    wb *= scale
    wb = np.clip(wb, 0, 255).astype(np.uint8)
    return wb

def ecc_homography(template_gray, candidate_gray, iterations=200, epsilon=1e-6):
    """Dense alignment fallback usando ECC."""
    warp_matrix = np.eye(3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, epsilon)
    template = template_gray.astype(np.float32) / 255.0
    candidate = candidate_gray.astype(np.float32) / 255.0
    try:
        _, warp_matrix = cv2.findTransformECC(
            template, candidate, warp_matrix, cv2.MOTION_HOMOGRAPHY, criteria, None, 5
        )
    except cv2.error as exc:
        raise RuntimeError(f"ECC optimization failed: {exc}") from exc
    return warp_matrix

def align_image(image_path, reference_path=None, crop_box=None):
    """Allinea un'immagine a un riferimento (come preprocess_alignment.py)."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    
    # Se non c'è riferimento, assume che l'immagine sia già allineata
    if reference_path is None:
        normalized = normalize_lighting(image)
        if crop_box:
            xmin, ymin, xmax, ymax = crop_box
            normalized = normalized[ymin:ymax, xmin:xmax]
        return normalized
    
    # Allinea usando omografia (come preprocess_alignment.py)
    template = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
    if template is None:
        raise FileNotFoundError(f"Could not read reference: {image_path}")
    
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Prova prima con ORB, poi con ECC se fallisce
    try:
        H = detect_homography(template_gray, gray)
    except RuntimeError:
        # Fallback a ECC
        H = ecc_homography(template_gray, gray)
    
    # Warp l'immagine
    warped = cv2.warpPerspective(image, H, (template.shape[1], template.shape[0]))
    
    # Applica crop box se fornito (per immagini raw)
    if crop_box:
        xmin, ymin, xmax, ymax = crop_box
        warped = warped[ymin:ymax, xmin:xmax]
    
    # Normalizza illuminazione
    normalized = normalize_lighting(warped)
    return normalized

def normalize_roi(img: np.ndarray) -> np.ndarray:
    """
    Convert ROI to grayscale, apply CLAHE, then normalize to float32 [0, 1].
    Identica a crop_connectors.py per garantire coerenza con il training.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    normalized = equalized.astype(np.float32) / 255.0
    return normalized

def extract_connectors(aligned_image, rois, margin=8):
    """Estrae i connettori da un'immagine allineata (come crop_connectors.py)."""
    height, width = aligned_image.shape[:2]
    connectors = []
    
    for roi in rois:
        x_min, y_min, x_max, y_max = roi.to_pixel_box(width, height, margin=margin)
        crop_bgr = aligned_image[y_min:y_max, x_min:x_max].copy()
        
        # Normalizza come in crop_connectors.py (grayscale + CLAHE)
        normalized = normalize_roi(crop_bgr)
        
        # Converti a uint8 per salvare/visualizzare (come in crop_connectors.py)
        crop_normalized = (normalized * 255).astype(np.uint8)
        
        connectors.append({
            'name': roi.name,
            'crop': crop_normalized,  # Grayscale normalizzato (come in Data/connectors/)
            'crop_bgr': crop_bgr,  # Mantieni BGR per visualizzazione originale
            'bbox': (x_min, y_min, x_max, y_max)
        })
    
    return connectors

# ============================================================================
# FUNZIONI CLASSIFICAZIONE
# ============================================================================

def preprocess_image(image_path, device=None):
    """Preprocessa immagine per classificatore OCCLUSION."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Carica immagine (può essere grayscale o RGB)
    image = Image.open(image_path)
    # Se è grayscale, converti in RGB (ripeti canale 3 volte)
    if image.mode == 'L':
        image = image.convert('RGB')
    elif image.mode != 'RGB':
        image = image.convert('RGB')
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    return image_tensor

def preprocess_image_for_ae(image_path, device=None):
    """Preprocessa immagine per autoencoder (LEGACY)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    # Carica immagine (può essere grayscale o RGB)
    image = Image.open(image_path)
    # Se è grayscale, converti in RGB (ripeti canale 3 volte)
    if image.mode == 'L':
        image = image.convert('RGB')
    elif image.mode != 'RGB':
        image = image.convert('RGB')
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    return image_tensor

def preprocess_image_for_efficientad(image_path, device=None):
    """Preprocessa immagine per EfficientAD-M (normalizzazione ImageNet)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Carica immagine (può essere grayscale o RGB)
    image = Image.open(image_path)
    # Se è grayscale, converti in RGB
    if image.mode == 'L':
        image = image.convert('RGB')
    elif image.mode != 'RGB':
        image = image.convert('RGB')
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    return image_tensor

def classify_connector(image_path, connector_name, occ_model, efficientad_models_dict, device=None, use_efficientad=True):
    """Classifica un connettore come OK, KO o OCCLUSION.
    
    Args:
        image_path: Path all'immagine o array numpy
        connector_name: Nome del connettore (conn1, conn2, ...)
        occ_model: Modello per classificazione OCCLUSION
        efficientad_models_dict: Dizionario con modelli EfficientAD-M {connector_name: (teacher, student, threshold)}
        device: Device PyTorch
        use_efficientad: Se True usa EfficientAD-M, altrimenti autoencoder (legacy)
    
    Returns:
        (label, score, heatmap, reconstructed): 
            - label: "OK", "KO" o "OCCLUSION"
            - score: Anomaly score (errore per autoencoder, differenza feature per EfficientAD-M)
            - heatmap: Array numpy [H, W] con heatmap (None se OCCLUSION)
            - reconstructed: None (non usato con EfficientAD-M)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # STEP 1: Verifica occlusione
    x_occ = preprocess_image(image_path, device)
    with torch.no_grad():
        logits = occ_model(x_occ)
        pred_vis = torch.argmax(logits, dim=1).item()
    
    if pred_vis == 0:  # OCCLUSION
        return "OCCLUSION", 0.0, None, None
    
    # STEP 2: Anomaly detection
    if connector_name not in efficientad_models_dict:
        raise ValueError(f"Modello non trovato per {connector_name}")
    
    if use_efficientad:
        # Usa EfficientAD-M
        teacher, student, threshold = efficientad_models_dict[connector_name]
        
        x_eff = preprocess_image_for_efficientad(image_path, device)
        with torch.no_grad():
            # Feature Teacher e Student
            t_feat = teacher(x_eff)
            s_feat = student(x_eff)
            
            # Differenza feature (anomaly map)
            diff = (t_feat - s_feat) ** 2
            # Media sui canali: [B, C, H, W] -> [B, H, W]
            amap = diff.mean(dim=1)
            # Score immagine = max su tutta la feature map
            score = amap.flatten(1).max(1)[0].cpu().item()
            
            # Heatmap: upsample amap alla dimensione originale (128x128)
            # amap è [B, Hf, Wf] dove Hf=H/32, Wf=W/32 (circa 4x4 per 128x128 input)
            amap_np = amap.squeeze(0).cpu().numpy()  # [Hf, Wf]
            # Upsample a 128x128
            import cv2
            heatmap = cv2.resize(amap_np, (128, 128), interpolation=cv2.INTER_LINEAR)
        
        if score > threshold:
            return "KO", score, heatmap, None
        else:
            return "OK", score, heatmap, None
    else:
        # Legacy: usa autoencoder
        ae_model, threshold = efficientad_models_dict[connector_name]
        
        x_ae = preprocess_image_for_ae(image_path, device)
        with torch.no_grad():
            reconstructed = ae_model(x_ae)
            criterion = nn.MSELoss(reduction='none')
            batch_errors = criterion(reconstructed, x_ae)
            pixel_errors = batch_errors.mean(dim=1)
            error = pixel_errors.mean().item()
            heatmap = pixel_errors.squeeze(0).cpu().numpy()
            reconstructed_np = reconstructed.squeeze(0).cpu().numpy()
            reconstructed_np = np.transpose(reconstructed_np, (1, 2, 0))
            reconstructed_np = (reconstructed_np * 255).astype(np.uint8)
        
        if error > threshold:
            return "KO", error, heatmap, reconstructed_np
        else:
            return "OK", error, heatmap, reconstructed_np

# ============================================================================
# CONFIGURAZIONE THRESHOLD MULTIPLIER
# ============================================================================

# Moltiplicatori per threshold per connettore (modificabili)
# Valore di default: 1.0 (usa threshold originale)
# Per conn3: 3.5 / 2.5 = 1.4 (aumenta threshold del 40%)
THRESHOLD_MULTIPLIERS = {
    'conn1': 1.0,
    'conn2': 1.0,
    'conn3': 1.4,  # 3.5 / 2.5 = 1.4 (threshold originale era calcolato con 2.5)
    'conn4': 1.0,
    'conn5': 1.0,
    'conn6': 1.0,
    'conn7': 1.2,
    'conn8': 1.4,
    'conn9': 1.0,
}

# ============================================================================
# INTERFACCIA GRAFICA
# ============================================================================

class BekoDetectionSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("BEKO DETECTION SYSTEM")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#ffffff')
        
        # Variabili
        self.image_path = None
        self.aligned_image = None
        self.connectors = []
        self.results = []
        self.occ_model = None
        self.ae_models = {}  # Dizionario: connector_name -> (model, threshold) - LEGACY
        self.efficientad_models = {}  # Dizionario: connector_name -> (teacher, student, threshold)
        self.use_efficientad = True  # Usa EfficientAD-M invece di autoencoder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Paths
        self.project_root = Path(__file__).parent.parent
        self.models_dir = self.project_root / "Training" / "models"
        self.roi_config_path = self.project_root / "Codice" / "roi_config.json"
        
        # Paths logo
        self.logo_polimi_path = self.project_root / "Logo polimi.png"
        self.logo_beko_path = self.project_root / "Logo beko.jpg"
        
        # Paths per riferimento e crop box (per allineamento)
        # Cerca automaticamente un'immagine di riferimento
        self.reference_path = None
        self.crop_box = None  # (302, 288, 1883, 942) per immagini raw
        
        # Cerca riferimento in Data/aligned_top o Data/TOP 1
        aligned_dir = self.project_root / "Data" / "aligned_top"
        top1_dir = self.project_root / "Data" / "TOP 1"
        
        if aligned_dir.exists():
            # Se esiste aligned_top, usa la prima immagine come riferimento (già allineata)
            refs = list(aligned_dir.glob("*.png"))
            if refs:
                self.reference_path = refs[0]
                self.crop_box = None  # Già allineata, non serve crop
        elif top1_dir.exists():
            # Se esiste TOP 1, cerca l'immagine di riferimento standard
            ref_path = top1_dir / "20251106131917_TOP.png"
            if ref_path.exists():
                self.reference_path = ref_path
                self.crop_box = (302, 288, 1883, 942)  # Crop box per immagini raw
        
        self.setup_ui()
        self.load_models()
        # Mostra informazioni sul threshold
        self.print_threshold_info()
    
    def setup_ui(self):
        # Header con logo
        header_frame = tk.Frame(self.root, bg='#ffffff', height=120)
        header_frame.pack(fill=tk.X, padx=0, pady=0)
        header_frame.pack_propagate(False)
        
        # Container per logo e titolo
        header_content = tk.Frame(header_frame, bg='#ffffff')
        header_content.pack(fill=tk.BOTH, expand=True, padx=30, pady=15)
        
        # Logo sinistra (Politecnico)
        if self.logo_polimi_path.exists():
            try:
                logo_polimi_img = Image.open(self.logo_polimi_path)
                logo_polimi_img = logo_polimi_img.resize((200, 65), Image.Resampling.LANCZOS)
                logo_polimi_photo = ImageTk.PhotoImage(logo_polimi_img)
                logo_polimi_label = tk.Label(
                    header_content,
                    image=logo_polimi_photo,
                    bg='#ffffff'
                )
                logo_polimi_label.image = logo_polimi_photo  # Keep a reference
                logo_polimi_label.pack(side=tk.LEFT, padx=(0, 20))
            except Exception as e:
                print(f"⚠️  Errore caricamento logo Politecnico: {e}")
        
        # Titolo centrale
        title_frame = tk.Frame(header_content, bg='#ffffff')
        title_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        
        title_label = tk.Label(
            title_frame,
            text="BEKO DETECTION SYSTEM",
            font=("Arial", 32, "bold"),
            bg='#ffffff',
            fg='#1a1a1a'
        )
        title_label.pack()
        
        group_label = tk.Label(
            title_frame,
            text="Group 3 • Ready to Use Software",
            font=("Arial", 20, "bold"),
            bg='#ffffff',
            fg='#007bff'
        )
        group_label.pack(pady=(5, 0))
        
        subtitle_label = tk.Label(
            title_frame,
            text="PCB Connector Quality Control",
            font=("Arial", 14),
            bg='#ffffff',
            fg='#666666'
        )
        subtitle_label.pack(pady=(5, 10))
        
        # Logo destra (Beko)
        if self.logo_beko_path.exists():
            try:
                logo_beko_img = Image.open(self.logo_beko_path)
                # Mantieni aspect ratio
                aspect = logo_beko_img.width / logo_beko_img.height
                new_height = 65
                new_width = int(new_height * aspect)
                logo_beko_img = logo_beko_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logo_beko_photo = ImageTk.PhotoImage(logo_beko_img)
                logo_beko_label = tk.Label(
                    header_content,
                    image=logo_beko_photo,
                    bg='#ffffff'
                )
                logo_beko_label.image = logo_beko_photo  # Keep a reference
                logo_beko_label.pack(side=tk.RIGHT, padx=(20, 0))
            except Exception as e:
                print(f"⚠️  Errore caricamento logo Beko: {e}")
        
        # Separatore
        separator = tk.Frame(self.root, bg='#e0e0e0', height=2)
        separator.pack(fill=tk.X, padx=0, pady=0)
        
        # Main container
        main_frame = tk.Frame(self.root, bg='#f5f5f5')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left panel - Upload e Statistiche
        left_panel = tk.Frame(main_frame, bg='#ffffff', width=380)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 20))
        left_panel.pack_propagate(False)
        
        # Upload area
        upload_frame = tk.Frame(left_panel, bg='#f8f9fa', relief=tk.FLAT, bd=0)
        upload_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        upload_title = tk.Label(
            upload_frame,
            text="Load Image",
            font=("Arial", 14, "bold"),
            bg='#f8f9fa',
            fg='#1a1a1a'
        )
        upload_title.pack(pady=(15, 10))
        
        upload_inner = tk.Frame(upload_frame, bg='#ffffff', relief=tk.SOLID, bd=2)
        upload_inner.pack(fill=tk.BOTH, expand=True, padx=15, pady=15, ipady=40)
        
        upload_label = tk.Label(
            upload_inner,
            text="📁 Drag image here\n\nor\n\n🖱️  Click to select",
            font=("Arial", 13),
            bg='#ffffff',
            fg='#666666',
            justify=tk.CENTER,
            cursor='hand2'
        )
        upload_label.pack(expand=True)
        
        # Bind drag & drop
        upload_inner.drop_target_register(DND_FILES)
        upload_inner.dnd_bind('<<Drop>>', self.on_drop)
        upload_label.bind("<Button-1>", self.on_click_select)
        upload_inner.bind("<Button-1>", self.on_click_select)
        
        # Status
        status_frame = tk.Frame(left_panel, bg='#ffffff')
        status_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.status_label = tk.Label(
            status_frame,
            text="✅ System ready",
            font=("Arial", 11),
            bg='#ffffff',
            fg='#28a745',
            anchor='w'
        )
        self.status_label.pack(fill=tk.X, pady=(0, 5))
        
        # Info riferimento
        ref_info = f"📎 Reference: {Path(self.reference_path).name if self.reference_path else 'None (assumes aligned)'}"
        self.ref_label = tk.Label(
            status_frame,
            text=ref_info,
            font=("Arial", 9),
            bg='#ffffff',
            fg='#999999',
            anchor='w'
        )
        self.ref_label.pack(fill=tk.X)
        
        # Process button
        self.process_btn = tk.Button(
            left_panel,
            text="🔍 ANALYZE IMAGE",
            font=("Arial", 14, "bold"),
            bg='#007bff',
            fg='#ffffff',
            command=self.process_image,
            state=tk.DISABLED,
            pady=12,
            relief=tk.FLAT,
            bd=0,
            cursor='hand2',
            activebackground='#0056b3',
            activeforeground='#ffffff'
        )
        self.process_btn.pack(fill=tk.X, pady=(0, 15))
        
        # Statistics
        stats_frame = tk.Frame(left_panel, bg='#f8f9fa', relief=tk.FLAT, bd=0)
        stats_frame.pack(fill=tk.BOTH, expand=True)
        
        stats_title = tk.Label(
            stats_frame,
            text="📊 Statistics",
            font=("Arial", 14, "bold"),
            bg='#f8f9fa',
            fg='#1a1a1a',
            anchor='w'
        )
        stats_title.pack(pady=(0, 10), padx=15, fill=tk.X)
        
        self.stats_text = tk.Text(
            stats_frame,
            font=("Courier", 10),
            bg='#ffffff',
            fg='#1a1a1a',
            wrap=tk.WORD,
            state=tk.DISABLED,
            relief=tk.FLAT,
            bd=0,
            padx=15,
            pady=15
        )
        self.stats_text.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
        
        # Right panel - Results
        right_panel = tk.Frame(main_frame, bg='#ffffff')
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Matplotlib figure con sfondo bianco (più grande per immagini più grandi)
        self.fig = plt.figure(figsize=(18, 14), facecolor='#ffffff', dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initial empty plot
        ax = self.fig.add_subplot(111, facecolor='#ffffff')
        ax.text(0.5, 0.5, "Load an image to start analysis", 
                ha='center', va='center', fontsize=16, color='#999999',
                transform=ax.transAxes)
        ax.axis('off')
        self.canvas.draw()
    
    def load_models(self):
        """Carica i modelli addestrati (EfficientAD-M o autoencoder legacy)."""
        try:
            self.status_label.config(text="⏳ Loading models...", fg='#ffc107')
            self.root.update()
            
            # Carica occlusion model
            occ_path = self.models_dir / "occlusion_cnn.pth"
            if not occ_path.exists():
                raise FileNotFoundError(f"Modello non trovato: {occ_path}")
            
            self.occ_model = OcclusionCNN().to(self.device)
            self.occ_model.load_state_dict(torch.load(occ_path, map_location=self.device))
            self.occ_model.eval()
            
            # Prova prima a caricare EfficientAD-M
            connectors = [f"conn{i}" for i in range(1, 10)]
            loaded_eff = 0
            
            for connector_name in connectors:
                student_path = self.models_dir / f"efficientad_student_{connector_name}.pth"
                threshold_path = self.models_dir / f"efficientad_threshold_{connector_name}.npy"
                
                if student_path.exists() and threshold_path.exists():
                    # Carica Teacher (sempre lo stesso, pre-addestrato)
                    teacher = Teacher().to(self.device)
                    teacher.eval()
                    
                    # Carica Student
                    student = Student().to(self.device)
                    student.load_state_dict(torch.load(student_path, map_location=self.device))
                    student.eval()
                    
                    # Carica threshold
                    threshold = np.load(threshold_path)
                    
                    # Applica moltiplicatore se configurato
                    multiplier = THRESHOLD_MULTIPLIERS.get(connector_name, 1.0)
                    if multiplier != 1.0:
                        threshold = threshold * multiplier
                        print(f"  ⚙️  Threshold {connector_name} moltiplicato per {multiplier:.2f} (nuovo: {threshold:.6f})")
                    
                    self.efficientad_models[connector_name] = (teacher, student, threshold)
                    loaded_eff += 1
                else:
                    print(f"⚠️  Modello EfficientAD non trovato per {connector_name}")
            
            if loaded_eff > 0:
                # Usa EfficientAD-M
                self.use_efficientad = True
                self.status_label.config(text=f"✅ Models loaded ({loaded_eff} EfficientAD-M)", fg='#28a745')
                print(f"✅ Caricati {loaded_eff} modelli EfficientAD-M")
            else:
                # Fallback: prova autoencoder (legacy)
                print("⚠️  EfficientAD-M non trovato, provo autoencoder...")
                self.use_efficientad = False
                loaded_ae = 0
                
                for connector_name in connectors:
                    ae_path = self.models_dir / f"ae_conv_{connector_name}.pth"
                    threshold_path = self.models_dir / f"ae_threshold_{connector_name}.npy"
                    
                    if ae_path.exists() and threshold_path.exists():
                        model = ConvAE().to(self.device)
                        model.load_state_dict(torch.load(ae_path, map_location=self.device))
                        model.eval()
                        threshold = np.load(threshold_path)
                        self.ae_models[connector_name] = (model, threshold)
                        loaded_ae += 1
                
                if loaded_ae == 0:
                    # Fallback: modello unico
                    ae_path = self.models_dir / "ae_conv.pth"
                    threshold_path = self.models_dir / "ae_threshold.npy"
                    if ae_path.exists() and threshold_path.exists():
                        print("⚠️  Usando modello unico (vecchio formato)")
                        model = ConvAE().to(self.device)
                        model.load_state_dict(torch.load(ae_path, map_location=self.device))
                        model.eval()
                        threshold = np.load(threshold_path)
                        for connector_name in connectors:
                            self.ae_models[connector_name] = (model, threshold)
                        loaded_ae = len(connectors)
                    else:
                        raise FileNotFoundError("Nessun modello trovato (né EfficientAD-M né autoencoder)")
                
                self.status_label.config(text=f"✅ Models loaded ({loaded_ae} AE legacy)", fg='#28a745')
                print(f"✅ Caricati {loaded_ae} modelli AE (legacy)")
        except Exception as e:
            messagebox.showerror("Errore", f"Errore caricamento modelli:\n{e}")
            self.status_label.config(text="❌ Error loading models", fg='#dc3545')
    
    def print_threshold_info(self):
        """Stampa informazioni sui thresholds per debug."""
        print(f"\n{'='*60}")
        print(f"🔍 INFORMAZIONI SUL SISTEMA DI CLASSIFICAZIONE")
        print(f"{'='*60}")
        
        if self.use_efficientad and self.efficientad_models:
            print(f"\n📊 THRESHOLDS EFFICIENTAD-M (per connettore):")
            for conn_name, (_, _, threshold) in sorted(self.efficientad_models.items()):
                print(f"  {conn_name}: {threshold:.6f}")
        elif self.ae_models:
            print(f"\n📊 THRESHOLDS AUTOENCODER (per connettore) - LEGACY:")
            for conn_name, (_, threshold) in sorted(self.ae_models.items()):
                print(f"  {conn_name}: {threshold:.6f}")
        
        print(f"{'='*60}\n")
    
    def on_drop(self, event):
        """Gestisce il drag & drop."""
        files = self.root.tk.splitlist(event.data)
        if files:
            self.image_path = files[0].strip('{}')
            self.status_label.config(text=f"📁 Image loaded: {Path(self.image_path).name}", fg='#28a745')
            self.process_btn.config(state=tk.NORMAL)
    
    def on_click_select(self, event):
        """Apre file dialog per selezionare immagine."""
        file_path = filedialog.askopenfilename(
            title="Select PCB Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")]
        )
        if file_path:
            self.image_path = file_path
            self.status_label.config(text=f"📁 Image loaded: {Path(self.image_path).name}", fg='#28a745')
            self.process_btn.config(state=tk.NORMAL)
    
    def process_image(self):
        """Processa l'immagine in un thread separato."""
        if not self.image_path:
            return
        
        self.process_btn.config(state=tk.DISABLED)
        self.status_label.config(text="⏳ Processing...", fg='#ffc107')
        self.root.update()
        
        # Esegui in thread per non bloccare UI
        thread = threading.Thread(target=self._process_image_thread)
        thread.daemon = True
        thread.start()
    
    def _process_image_thread(self):
        """Thread per processare l'immagine."""
        try:
            # 1. Allinea immagine (usa riferimento e crop box se disponibili)
            self.root.after(0, lambda: self.status_label.config(text="⏳ Aligning image...", fg='#ffc107'))
            
            # Determina se usare riferimento o assumere già allineata
            ref_path = self.reference_path if self.reference_path and self.reference_path.exists() else None
            crop = self.crop_box
            
            if ref_path:
                self.root.after(0, lambda: self.status_label.config(
                    text=f"⏳ Aligning with reference: {Path(ref_path).name}...", fg='#ffc107'))
            
            self.aligned_image = align_image(self.image_path, reference_path=ref_path, crop_box=crop)
            
            # 2. Carica ROI config
            if not self.roi_config_path.exists():
                raise FileNotFoundError(f"ROI config not found: {self.roi_config_path}")
            rois = load_roi_config(self.roi_config_path)
            
            # 3. Estrai connettori
            self.root.after(0, lambda: self.status_label.config(text="⏳ Extracting connectors...", fg='#ffc107'))
            self.connectors = extract_connectors(self.aligned_image, rois, margin=8)
            
            # 4. Classifica connettori
            self.root.after(0, lambda: self.status_label.config(text="⏳ Classifying connectors...", fg='#ffc107'))
            temp_dir = Path(tempfile.mkdtemp())
            self.results = []
            
            for i, conn in enumerate(self.connectors):
                self.root.after(0, lambda idx=i: self.status_label.config(
                    text=f"⏳ Classifying {idx+1}/9...", fg='#ffc107'))
                
                temp_path = temp_dir / f"{conn['name']}.png"
                # Salva crop normalizzato (grayscale) come in Data/connectors/
                cv2.imwrite(str(temp_path), conn['crop'])
                
                # Usa EfficientAD-M o autoencoder (legacy)
                if self.use_efficientad:
                    label, score, heatmap, reconstructed = classify_connector(
                        str(temp_path), conn['name'], self.occ_model, 
                        self.efficientad_models, self.device, use_efficientad=True
                    )
                    threshold = self.efficientad_models.get(conn['name'], (None, None, None))[2] if conn['name'] in self.efficientad_models else None
                    error = score  # Per compatibilità con il codice esistente
                else:
                    label, error, heatmap, reconstructed = classify_connector(
                        str(temp_path), conn['name'], self.occ_model, 
                        self.ae_models, self.device, use_efficientad=False
                    )
                    threshold = self.ae_models.get(conn['name'], (None, None))[1] if conn['name'] in self.ae_models else None
                
                self.results.append({
                    'name': conn['name'],
                    'crop': conn['crop'],
                    'bbox': conn['bbox'],
                    'label': label,
                    'error': error,
                    'heatmap': heatmap,
                    'reconstructed': reconstructed,
                    'threshold': threshold
                })
            
            # 5. Visualizza risultati
            self.root.after(0, self.visualize_results)
            self.root.after(0, lambda: self.status_label.config(text="✅ Analysis completed!", fg='#28a745'))
            self.root.after(0, lambda: self.process_btn.config(state=tk.NORMAL))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Errore", f"Errore durante l'elaborazione:\n{e}"))
            self.root.after(0, lambda: self.status_label.config(text="❌ Error", fg='#dc3545'))
            self.root.after(0, lambda: self.process_btn.config(state=tk.NORMAL))
    
    def visualize_results(self):
        """Visualizza i risultati - solo i 9 connettori con heatmap migliorata."""
        self.fig.clear()
        self.fig.set_facecolor('#ffffff')
        
        # Colori moderni e puliti
        COLORS = {
            'OK': '#28a745',        # Verde
            'KO': '#dc3545',        # Rosso
            'OCCLUSION': '#ffc107'  # Giallo/Ambra
        }
        
        # Colormap personalizzata per heatmap: giallo (normale) -> rosso (anomalo)
        # Giallo chiaro per zone normali, rosso intenso per zone anomale
        colors_heatmap = ['#ffff00', '#ffcc00', '#ff9900', '#ff6600', '#ff3300', '#cc0000']
        n_bins = 256
        cmap_heatmap = LinearSegmentedColormap.from_list('anomaly', colors_heatmap, N=n_bins)
        
        # Grid 3x3 con solo i connettori
        for idx, r in enumerate(self.results):
            ax = self.fig.add_subplot(3, 3, idx + 1)
            ax.set_facecolor('#ffffff')
            
            # I crop sono già in grayscale
            if len(r['crop'].shape) == 2:
                ax.imshow(r['crop'], cmap='gray', vmin=0, vmax=255)
            else:
                crop_rgb = cv2.cvtColor(r['crop'], cv2.COLOR_BGR2RGB)
                ax.imshow(crop_rgb)
            
            # Heatmap migliorata - solo per OK/KO, non OCCLUSION
            if r['heatmap'] is not None:
                # Resize heatmap se necessario
                h, w = r['crop'].shape[:2]
                if r['heatmap'].shape != (h, w):
                    # Mantieni i valori originali, non normalizzare prima del resize
                    heatmap_resized = cv2.resize(r['heatmap'], (w, h), interpolation=cv2.INTER_LINEAR)
                else:
                    heatmap_resized = r['heatmap'].copy()
                
                # Normalizza heatmap basandosi sul threshold
                threshold = r.get('threshold')
                if threshold is not None and threshold > 0:
                    # Usa threshold come punto di riferimento per la normalizzazione
                    # Valori sotto threshold/2 = giallo (normale)
                    # Valori sopra threshold = rosso (anomalo)
                    # Valori tra threshold/2 e threshold = gradiente giallo->arancione->rosso
                    
                    # Normalizza: threshold corrisponde a ~0.6 nella scala 0-1
                    # Questo permette di vedere sia zone normali (gialle) che anomale (rosse)
                    vmax = threshold * 1.5  # Soglia massima: 1.5x threshold per vedere anche valori alti
                    vmin = 0
                    
                    # Normalizza la heatmap
                    heatmap_norm = np.clip(heatmap_resized / vmax, 0, 1)
                    
                    # Applica una funzione per enfatizzare le zone anomale
                    # Zone normali (basse) rimangono gialle, zone anomale (alte) diventano rosse
                    # Usa una funzione per rendere più visibili le differenze
                    heatmap_norm = np.power(heatmap_norm, 0.8)  # Leggera enfatizzazione
                else:
                    # Fallback: normalizzazione min-max
                    vmax = heatmap_resized.max()
                    vmin = heatmap_resized.min()
                    if vmax > vmin:
                        heatmap_norm = (heatmap_resized - vmin) / (vmax - vmin)
                    else:
                        heatmap_norm = np.zeros_like(heatmap_resized)
                
                # Mostra SEMPRE la heatmap, non mascherarla
                # Alpha più alto per zone KO, più basso per OK
                alpha = 0.5 if r['label'] == 'OK' else 0.65
                im = ax.imshow(heatmap_norm, cmap=cmap_heatmap, alpha=alpha, interpolation='bilinear', vmin=0, vmax=1)
                
                # Aggiungi contorni per evidenziare le zone più anomale (solo se c'è threshold)
                if threshold is not None and threshold > 0:
                    # Contorni per zone sopra 70% del threshold (zone molto anomale)
                    contour_threshold = threshold * 0.7
                    # Normalizza il threshold per i contorni
                    contour_level_norm = contour_threshold / (threshold * 1.5)
                    if contour_level_norm <= 1.0 and heatmap_resized.max() >= contour_threshold:
                        # Trova i contorni nella heatmap originale
                        contour_mask = heatmap_resized >= contour_threshold
                        if contour_mask.any():
                            # Crea contorni dalla maschera usando XOR invece di sottrazione
                            dilated = ndimage.binary_dilation(contour_mask)
                            contours_data = np.logical_xor(dilated, contour_mask).astype(np.float32)
                            if contours_data.any():
                                ax.contour(contours_data, levels=[0.5], colors=['darkred'], linewidths=2, alpha=0.9)
            
            ax.axis('off')
            
            # Colore per label
            color_hex = COLORS[r['label']]
            
            # Bordo colorato più spesso e visibile
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(color_hex)
                spine.set_linewidth(3)
            
            # Label sopra il crop (più piccola e discreta)
            label_text = r['label']
            threshold = r.get('threshold')
            if r['error'] > 0 and threshold is not None:
                if r['label'] == 'KO':
                    label_text += f"\n{r['error']:.3f}"
                elif r['label'] == 'OK':
                    label_text += f"\n{r['error']:.3f}"
            
            # Label con sfondo colorato (più piccola)
            ax.text(0.5, 0.97, label_text,
                   transform=ax.transAxes,
                   fontsize=7, fontweight='bold',
                   color='white',
                   ha='center', va='top',
                   bbox=dict(boxstyle='round,pad=0.25',
                           facecolor=color_hex,
                           edgecolor='white',
                           linewidth=1,
                           alpha=0.85))
            
            # Nome connettore in basso (più piccolo)
            ax.text(0.5, 0.03, r['name'],
                   transform=ax.transAxes,
                   fontsize=7,
                   color='#1a1a1a',
                   ha='center', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.2',
                           facecolor='white',
                           edgecolor='#cccccc',
                           linewidth=0.8,
                           alpha=0.85))
        
        plt.tight_layout(pad=2.0, rect=[0, 0, 1, 0.98])
        self.canvas.draw()
        
        # Aggiorna statistiche
        self.update_statistics()
    
    def update_statistics(self):
        """Aggiorna le statistiche."""
        ok_count = sum(1 for r in self.results if r['label'] == 'OK')
        ko_count = sum(1 for r in self.results if r['label'] == 'KO')
        occ_count = sum(1 for r in self.results if r['label'] == 'OCCLUSION')
        
        stats_text = f"✅ OK:        {ok_count}/9\n"
        stats_text += f"❌ KO:        {ko_count}/9\n"
        stats_text += f"⚠️  OCCLUSION: {occ_count}/9\n\n"
        stats_text += "=" * 35 + "\n\n"
        
        for r in self.results:
            threshold = r.get('threshold')
            stats_text += f"{r['name']}: {r['label']}\n"
            if r['error'] > 0:
                stats_text += f"  Score: {r['error']:.6f}\n"
                if threshold is not None:
                    stats_text += f"  Threshold: {threshold:.6f}\n"
                if r['heatmap'] is not None:
                    stats_text += f"  Heatmap: max={r['heatmap'].max():.4f}, mean={r['heatmap'].mean():.4f}\n"
            stats_text += "\n"
        
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats_text)
        self.stats_text.config(state=tk.DISABLED)

def main():
    # Verifica dipendenze
    try:
        import tkinterdnd2
    except ImportError:
        print("⚠️  tkinterdnd2 non installato. Installazione...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tkinterdnd2"])
        import tkinterdnd2
    
    root = TkinterDnD.Tk()
    app = BekoDetectionSystem(root)
    root.mainloop()

if __name__ == "__main__":
    main()
