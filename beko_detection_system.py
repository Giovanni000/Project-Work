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
from PIL import Image, ImageTk
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle
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
    """Autoencoder per anomaly detection."""
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
        raise FileNotFoundError(f"Could not read reference: {reference_path}")
    
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
    """Preprocessa immagine per autoencoder."""
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

def classify_connector(image_path, connector_name, occ_model, ae_models_dict, device=None):
    """Classifica un connettore come OK, KO o OCCLUSION.
    
    Returns:
        (label, error, heatmap, reconstructed): label è "OK", "KO" o "OCCLUSION", 
                                                error è l'errore medio di ricostruzione,
                                                heatmap è un array numpy [H, W] con errore per pixel (None se OCCLUSION),
                                                reconstructed è l'immagine ricostruita [H, W, 3] (None se OCCLUSION)
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
    
    # STEP 2: Anomaly detection (usa il modello specifico per questo connettore)
    if connector_name not in ae_models_dict:
        raise ValueError(f"Modello non trovato per {connector_name}")
    
    ae_model, threshold = ae_models_dict[connector_name]
    
    x_ae = preprocess_image_for_ae(image_path, device)
    with torch.no_grad():
        reconstructed = ae_model(x_ae)
        # Calcola errore come nel training: MSE con reduction='none', poi media su [C, H, W]
        # Questo è identico a come viene calcolato nel calculate_threshold
        criterion = nn.MSELoss(reduction='none')
        batch_errors = criterion(reconstructed, x_ae)
        # Media su canali RGB: [1, 3, H, W] -> [1, H, W]
        pixel_errors = batch_errors.mean(dim=1)  # Media sui canali RGB
        # Media spaziale per errore totale: [1, H, W] -> scalare
        error = pixel_errors.mean().item()
        # Heatmap: [1, H, W] -> [H, W] (numpy)
        heatmap = pixel_errors.squeeze(0).cpu().numpy()
        # Immagine ricostruita: [1, 3, H, W] -> [H, W, 3] (numpy, uint8)
        reconstructed_np = reconstructed.squeeze(0).cpu().numpy()
        reconstructed_np = np.transpose(reconstructed_np, (1, 2, 0))  # [H, W, 3]
        reconstructed_np = (reconstructed_np * 255).astype(np.uint8)
    
    if error > threshold:
        return "KO", error, heatmap, reconstructed_np
    else:
        return "OK", error, heatmap, reconstructed_np

# ============================================================================
# INTERFACCIA GRAFICA
# ============================================================================

class BekoDetectionSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("BEKO DETECTION SYSTEM")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2b2b2b')
        
        # Variabili
        self.image_path = None
        self.aligned_image = None
        self.connectors = []
        self.results = []
        self.occ_model = None
        self.ae_models = {}  # Dizionario: connector_name -> (model, threshold)
        self.threshold = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Paths
        self.project_root = Path(__file__).parent.parent
        self.models_dir = self.project_root / "Training" / "models"
        self.roi_config_path = self.project_root / "Codice" / "roi_config.json"
        
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
        # Header
        header_frame = tk.Frame(self.root, bg='#1e1e1e', height=80)
        header_frame.pack(fill=tk.X, padx=0, pady=0)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame,
            text="BEKO DETECTION SYSTEM",
            font=("Arial", 28, "bold"),
            bg='#1e1e1e',
            fg='#00ff00'
        )
        title_label.pack(pady=20)
        
        # Main container
        main_frame = tk.Frame(self.root, bg='#2b2b2b')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Upload
        left_panel = tk.Frame(main_frame, bg='#2b2b2b', width=400)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # Upload area
        upload_frame = tk.Frame(left_panel, bg='#3b3b3b', relief=tk.RAISED, bd=2)
        upload_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        upload_label = tk.Label(
            upload_frame,
            text="📁 Trascina immagine qui\n\noppure\n\n🖱️  Clicca per selezionare",
            font=("Arial", 16),
            bg='#3b3b3b',
            fg='#ffffff',
            pady=60,
            justify=tk.CENTER
        )
        upload_label.pack(expand=True)
        
        # Bind drag & drop
        upload_frame.drop_target_register(DND_FILES)
        upload_frame.dnd_bind('<<Drop>>', self.on_drop)
        upload_label.bind("<Button-1>", self.on_click_select)
        upload_frame.bind("<Button-1>", self.on_click_select)
        
        # Status (più visibile)
        status_frame = tk.Frame(left_panel, bg='#2b2b2b')
        status_frame.pack(fill=tk.X, pady=8)
        
        self.status_label = tk.Label(
            status_frame,
            text="✅ Pronto",
            font=("Arial", 12, "bold"),
            bg='#2b2b2b',
            fg='#00ff00',
            anchor='w'
        )
        self.status_label.pack(fill=tk.X)
        
        # Info riferimento (più leggibile)
        ref_info = f"📎 Rif: {Path(self.reference_path).name if self.reference_path else 'Nessuno (assume allineata)'}"
        self.ref_label = tk.Label(
            status_frame,
            text=ref_info,
            font=("Arial", 9),
            bg='#2b2b2b',
            fg='#aaaaaa',
            anchor='w'
        )
        self.ref_label.pack(fill=tk.X, pady=(2, 0))
        
        # Process button (più visibile)
        self.process_btn = tk.Button(
            left_panel,
            text="🔍 ANALIZZA IMMAGINE",
            font=("Arial", 16, "bold"),
            bg='#00ff00',
            fg='#000000',
            command=self.process_image,
            state=tk.DISABLED,
            pady=15,
            relief=tk.RAISED,
            bd=3,
            cursor='hand2'
        )
        self.process_btn.pack(fill=tk.X, pady=12)
        
        # Statistics
        stats_frame = tk.Frame(left_panel, bg='#3b3b3b', relief=tk.RAISED, bd=2)
        stats_frame.pack(fill=tk.BOTH, expand=True)
        
        stats_title = tk.Label(
            stats_frame,
            text="📊 STATISTICHE",
            font=("Arial", 14, "bold"),
            bg='#3b3b3b',
            fg='white'
        )
        stats_title.pack(pady=12)
        
        self.stats_text = tk.Text(
            stats_frame,
            font=("Courier", 11),
            bg='#1e1e1e',
            fg='#ffffff',
            wrap=tk.WORD,
            state=tk.DISABLED,
            relief=tk.FLAT,
            bd=0,
            padx=12,
            pady=12
        )
        self.stats_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        # Right panel - Results
        right_panel = tk.Frame(main_frame, bg='#2b2b2b')
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Matplotlib figure (più grande per migliore leggibilità)
        self.fig = plt.figure(figsize=(14, 10), facecolor='#1e1e1e', dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initial empty plot
        ax = self.fig.add_subplot(111, facecolor='#2b2b2b')
        ax.text(0.5, 0.5, "Carica un'immagine per iniziare", 
                ha='center', va='center', fontsize=16, color='white',
                transform=ax.transAxes)
        ax.axis('off')
        self.canvas.draw()
    
    def load_models(self):
        """Carica i modelli addestrati (9 modelli AE, uno per connettore)."""
        try:
            self.status_label.config(text="Caricamento modelli...", fg='yellow')
            self.root.update()
            
            # Carica occlusion model
            occ_path = self.models_dir / "occlusion_cnn.pth"
            if not occ_path.exists():
                raise FileNotFoundError(f"Modello non trovato: {occ_path}")
            
            self.occ_model = OcclusionCNN().to(self.device)
            self.occ_model.load_state_dict(torch.load(occ_path, map_location=self.device))
            self.occ_model.eval()
            
            # Carica 9 modelli AE (uno per connettore)
            connectors = [f"conn{i}" for i in range(1, 10)]
            loaded = 0
            
            for connector_name in connectors:
                ae_path = self.models_dir / f"ae_conv_{connector_name}.pth"
                threshold_path = self.models_dir / f"ae_threshold_{connector_name}.npy"
                
                if ae_path.exists() and threshold_path.exists():
                    model = ConvAE().to(self.device)
                    model.load_state_dict(torch.load(ae_path, map_location=self.device))
                    model.eval()
                    threshold = np.load(threshold_path)
                    self.ae_models[connector_name] = (model, threshold)
                    loaded += 1
                else:
                    print(f"⚠️  Modello non trovato per {connector_name}")
            
            if loaded == 0:
                # Fallback: prova a caricare il vecchio modello unico
                ae_path = self.models_dir / "ae_conv.pth"
                threshold_path = self.models_dir / "ae_threshold.npy"
                if ae_path.exists() and threshold_path.exists():
                    print("⚠️  Usando modello unico (vecchio formato)")
                    model = ConvAE().to(self.device)
                    model.load_state_dict(torch.load(ae_path, map_location=self.device))
                    model.eval()
                    threshold = np.load(threshold_path)
                    # Usa lo stesso modello per tutti i connettori
                    for connector_name in connectors:
                        self.ae_models[connector_name] = (model, threshold)
                    loaded = len(connectors)
                else:
                    raise FileNotFoundError("Nessun modello AE trovato (né per connettore né unico)")
            
            self.status_label.config(text=f"✅ Modelli caricati ({loaded} AE)", fg='#00ff00')
            print(f"✅ Caricati {loaded} modelli AE")
        except Exception as e:
            messagebox.showerror("Errore", f"Errore caricamento modelli:\n{e}")
    
    def print_threshold_info(self):
        """Stampa informazioni sui thresholds per debug."""
        if self.ae_models:
            print(f"\n{'='*60}")
            print(f"🔍 INFORMAZIONI SUL SISTEMA DI CLASSIFICAZIONE")
            print(f"{'='*60}")
            print(f"\n📊 THRESHOLDS AUTOENCODER (per connettore):")
            for conn_name, (_, threshold) in sorted(self.ae_models.items()):
                print(f"  {conn_name}: {threshold:.6f}")
            print(f"\n📝 COME FUNZIONA:")
            print(f"  1. STEP 1 - Classificatore OCCLUSION vs VISIBLE:")
            print(f"     • Se OCCLUSION → ritorna 'OCCLUSION'")
            print(f"     • Se VISIBLE → procede a STEP 2")
            print(f"\n  2. STEP 2 - Autoencoder per Anomaly Detection:")
            print(f"     • Ogni connettore ha il suo autoencoder addestrato SOLO su immagini OK")
            print(f"     • Calcola errore di ricostruzione (MSE tra input e output)")
            print(f"     • Usa il threshold specifico del connettore per decidere OK/KO")
            print(f"\n⚠️  NOTA: Se molti KO vengono classificati come OK,")
            print(f"     i thresholds potrebbero essere troppo alti.")
            print(f"     Verifica gli errori di ricostruzione nella visualizzazione.")
            print(f"{'='*60}\n")
    
    def on_drop(self, event):
        """Gestisce il drag & drop."""
        files = self.root.tk.splitlist(event.data)
        if files:
            self.image_path = files[0].strip('{}')
            self.status_label.config(text=f"Immagine caricata: {Path(self.image_path).name}", fg='#00ff00')
            self.process_btn.config(state=tk.NORMAL)
    
    def on_click_select(self, event):
        """Apre file dialog per selezionare immagine."""
        file_path = filedialog.askopenfilename(
            title="Seleziona immagine PCB",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")]
        )
        if file_path:
            self.image_path = file_path
            self.status_label.config(text=f"Immagine caricata: {Path(self.image_path).name}", fg='#00ff00')
            self.process_btn.config(state=tk.NORMAL)
    
    def process_image(self):
        """Processa l'immagine in un thread separato."""
        if not self.image_path:
            return
        
        self.process_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Elaborazione in corso...", fg='yellow')
        self.root.update()
        
        # Esegui in thread per non bloccare UI
        thread = threading.Thread(target=self._process_image_thread)
        thread.daemon = True
        thread.start()
    
    def _process_image_thread(self):
        """Thread per processare l'immagine."""
        try:
            # 1. Allinea immagine (usa riferimento e crop box se disponibili)
            self.root.after(0, lambda: self.status_label.config(text="Allineamento immagine...", fg='yellow'))
            
            # Determina se usare riferimento o assumere già allineata
            ref_path = self.reference_path if self.reference_path and self.reference_path.exists() else None
            crop = self.crop_box
            
            if ref_path:
                self.root.after(0, lambda: self.status_label.config(
                    text=f"Allineamento con riferimento: {Path(ref_path).name}...", fg='yellow'))
            
            self.aligned_image = align_image(self.image_path, reference_path=ref_path, crop_box=crop)
            
            # 2. Carica ROI config
            if not self.roi_config_path.exists():
                raise FileNotFoundError(f"ROI config non trovato: {self.roi_config_path}")
            rois = load_roi_config(self.roi_config_path)
            
            # 3. Estrai connettori
            self.root.after(0, lambda: self.status_label.config(text="Estrazione connettori...", fg='yellow'))
            self.connectors = extract_connectors(self.aligned_image, rois, margin=8)
            
            # 4. Classifica connettori
            self.root.after(0, lambda: self.status_label.config(text="Classificazione connettori...", fg='yellow'))
            temp_dir = Path(tempfile.mkdtemp())
            self.results = []
            
            for i, conn in enumerate(self.connectors):
                self.root.after(0, lambda idx=i: self.status_label.config(
                    text=f"Classificazione {idx+1}/9...", fg='yellow'))
                
                temp_path = temp_dir / f"{conn['name']}.png"
                # Salva crop normalizzato (grayscale) come in Data/connectors/
                cv2.imwrite(str(temp_path), conn['crop'])
                
                label, error, heatmap, reconstructed = classify_connector(
                    str(temp_path), conn['name'], self.occ_model, self.ae_models, self.device
                )
                
                # Ottieni threshold per questo connettore
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
            self.root.after(0, lambda: self.status_label.config(text="✅ Analisi completata!", fg='#00ff00'))
            self.root.after(0, lambda: self.process_btn.config(state=tk.NORMAL))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Errore", f"Errore durante l'elaborazione:\n{e}"))
            self.root.after(0, lambda: self.status_label.config(text="❌ Errore", fg='red'))
            self.root.after(0, lambda: self.process_btn.config(state=tk.NORMAL))
    
    def visualize_results(self):
        """Visualizza i risultati - solo i 9 connettori."""
        self.fig.clear()
        self.fig.set_facecolor('#1e1e1e')
        
        # Colori più vivaci e contrastati
        COLORS = {
            'OK': (0, 255, 0),           # Verde brillante
            'KO': (255, 0, 0),           # Rosso brillante
            'OCCLUSION': (255, 200, 0)   # Giallo/arancione brillante
        }
        
        
        # Grid 3x3 con solo i connettori (più grandi)
        for idx, r in enumerate(self.results):
            ax = self.fig.add_subplot(3, 3, idx + 1)
            
            # I crop sono già in grayscale
            if len(r['crop'].shape) == 2:
                ax.imshow(r['crop'], cmap='gray', vmin=0, vmax=255)
            else:
                crop_rgb = cv2.cvtColor(r['crop'], cv2.COLOR_BGR2RGB)
                ax.imshow(crop_rgb)
            
            # Overlay heatmap se disponibile (solo per OK/KO, non OCCLUSION)
            if r['heatmap'] is not None:
                # Resize heatmap se necessario (dovrebbe essere 128x128, ma il crop potrebbe essere diverso)
                h, w = r['crop'].shape[:2]
                if r['heatmap'].shape != (h, w):
                    # Converti a uint8 per cv2.resize, poi riporta a float
                    heatmap_uint8 = (r['heatmap'] * 255).astype(np.uint8)
                    heatmap_resized = cv2.resize(heatmap_uint8, (w, h), interpolation=cv2.INTER_LINEAR)
                    heatmap_resized = heatmap_resized.astype(np.float32) / 255.0
                else:
                    heatmap_resized = r['heatmap']
                
                # Usa scala assoluta basata su threshold invece di normalizzazione per immagine
                # Questo permette di vedere le differenze reali tra OK e KO
                threshold = r.get('threshold')
                if threshold is not None:
                    # Scala: 0 = errore basso, threshold = errore medio-alto, 2*threshold = errore molto alto
                    vmax = max(threshold * 2, heatmap_resized.max())
                    vmin = 0
                else:
                    # Fallback: usa min-max se threshold non disponibile
                    vmax = heatmap_resized.max()
                    vmin = heatmap_resized.min()
                
                # Clip e normalizza per visualizzazione
                heatmap_clipped = np.clip(heatmap_resized, vmin, vmax)
                if vmax > vmin:
                    heatmap_norm = (heatmap_clipped - vmin) / (vmax - vmin)
                else:
                    heatmap_norm = np.zeros_like(heatmap_clipped)
                
                # Overlay heatmap con trasparenza (colori caldi: rosso/giallo = errore alto)
                # Usa alpha più alto per KO per evidenziare meglio le differenze
                alpha = 0.7 if r['label'] == 'KO' else 0.5
                im = ax.imshow(heatmap_norm, cmap='hot', alpha=alpha, interpolation='bilinear', vmin=0, vmax=1)
            
            ax.axis('off')
            ax.set_facecolor('#1e1e1e')
            
            # Colore per label
            color = COLORS[r['label']]
            color_hex = '#%02x%02x%02x' % color
            
            # Bordo colorato molto sottile (solo per indicare la classe)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(color_hex)
                spine.set_linewidth(1)  # Molto sottile
            
            # Label sopra il crop con errore e threshold info
            label_text = r['label']
            threshold = r.get('threshold')
            if r['error'] > 0:
                # Mostra errore e se è sopra/sotto threshold
                if threshold is not None:
                    if r['label'] == 'KO' and r['error'] > threshold:
                        label_text += f"\n({r['error']:.4f} > {threshold:.4f})"
                    elif r['label'] == 'OK' and r['error'] <= threshold:
                        label_text += f"\n({r['error']:.4f} ≤ {threshold:.4f})"
                    else:
                        label_text += f"\n({r['error']:.4f})"
                else:
                    label_text += f"\n({r['error']:.4f})"
            
            # Label piccola e discreta sopra
            ax.text(0.5, 0.98, label_text,
                   transform=ax.transAxes,
                   fontsize=8, fontweight='bold',
                   color='white',
                   ha='center', va='top',
                   bbox=dict(boxstyle='round,pad=0.2',
                           facecolor=color_hex,
                           edgecolor='white',
                           linewidth=0.5,
                           alpha=0.7))
            
            # Nome connettore piccolo in basso
            ax.text(0.5, 0.02, r['name'],
                   transform=ax.transAxes,
                   fontsize=8,
                   color='white',
                   ha='center', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.15',
                           facecolor='black',
                           edgecolor='none',
                           alpha=0.6))
        
        plt.tight_layout(pad=2.0, rect=[0, 0, 1, 0.96])
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
        stats_text += "=" * 40 + "\n"
        if hasattr(self, 'threshold'):
            stats_text += f"Threshold: {self.threshold:.6f}\n"
        stats_text += "=" * 40 + "\n\n"
        
        # Calcola statistiche heatmap per debug
        heatmap_errors = []
        for r in self.results:
            if r['heatmap'] is not None:
                heatmap_errors.append({
                    'name': r['name'],
                    'mean': r['heatmap'].mean(),
                    'max': r['heatmap'].max(),
                    'std': r['heatmap'].std(),
                    'label': r['label']
                })
        
        for r in self.results:
            threshold = r.get('threshold')
            stats_text += f"{r['name']}: {r['label']}"
            if r['error'] > 0:
                stats_text += f"\n  Errore medio: {r['error']:.6f}"
                if threshold is not None:
                    if r['error'] > threshold:
                        stats_text += f" > {threshold:.6f} (KO)"
                    else:
                        stats_text += f" ≤ {threshold:.6f} (OK)"
                    stats_text += f"\n  Threshold: {threshold:.6f}"
                # Aggiungi info heatmap se disponibile
                if r['heatmap'] is not None:
                    stats_text += f"\n  Heatmap: max={r['heatmap'].max():.4f}, mean={r['heatmap'].mean():.4f}, std={r['heatmap'].std():.4f}"
            stats_text += "\n\n"
        
        # Debug: confronta errori OK vs KO
        if heatmap_errors:
            ok_errors = [e['mean'] for e in heatmap_errors if e['label'] == 'OK']
            ko_errors = [e['mean'] for e in heatmap_errors if e['label'] == 'KO']
            if ok_errors and ko_errors:
                stats_text += "=" * 40 + "\n"
                stats_text += "DEBUG - Confronto Errori:\n"
                stats_text += f"  OK:  mean={np.mean(ok_errors):.6f}, max={np.max(ok_errors):.6f}\n"
                stats_text += f"  KO:  mean={np.mean(ko_errors):.6f}, max={np.max(ko_errors):.6f}\n"
                stats_text += f"  Diff: {np.mean(ko_errors) - np.mean(ok_errors):.6f}\n"
                stats_text += "=" * 40 + "\n"
        
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

