#!/usr/bin/env python3
"""
PCL ROI Annotator Tool

Interactive tool to manually define ROI (bounding box) for the PCL area of each connector.
For each connector, it shows an average image and lets you draw a rectangle ROI with the mouse.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import matplotlib.patches as patches

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

IMG_SIZE = 128
CSV_PATH = Path(__file__).parent / "data" / "dataset.csv"
OUTPUT_JSON = Path(__file__).parent / "pcl_roi_config.json"

# Path base per le immagini (se non sono nel CSV)
DATA_BASE = Path(__file__).parent.parent / "Data" / "connectors"

# ============================================================================
# FUNZIONI HELPER
# ============================================================================

def load_connector_images(connector_name, csv_path, data_base=None):
    """
    Carica le immagini per un connettore dal CSV.
    
    Args:
        connector_name: Nome del connettore (es. 'conn1')
        csv_path: Path al CSV del dataset
        data_base: Path base per le immagini (se i path nel CSV non sono assoluti)
    
    Returns:
        Lista di path alle immagini
    """
    df = pd.read_csv(csv_path)
    df_conn = df[df['connector_name'] == connector_name].copy()
    
    if len(df_conn) == 0:
        print(f"⚠️  Nessuna immagine trovata per {connector_name}")
        return []
    
    # Shuffle
    df_conn = df_conn.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Seleziona immagini: prime 15 + ultime 15 (o tutte se <= 30)
    if len(df_conn) <= 30:
        selected_df = df_conn
    else:
        first_15 = df_conn.head(15)
        last_15 = df_conn.tail(15)
        selected_df = pd.concat([first_15, last_15]).reset_index(drop=True)
    
    # Costruisci path immagini
    image_paths = []
    for idx, row in selected_df.iterrows():
        img_path = row.get('image_path', '')
        
        # Estrai filename dal path (anche se è un path Google Drive)
        if img_path:
            filename = Path(img_path).name
        else:
            filename = row.get('filename', '')
            if not filename:
                continue
        
        # Costruisci path locale: Data/connectors/connX/filename
        if data_base:
            local_path = data_base / connector_name / filename
        else:
            # Se non c'è data_base, prova il path originale
            local_path = Path(img_path) if img_path else None
        
        # Verifica esistenza
        if local_path and local_path.exists():
            image_paths.append(local_path)
        else:
            # Prova anche il path originale se diverso
            if img_path and Path(img_path).exists():
                image_paths.append(Path(img_path))
            else:
                print(f"  ⚠️  Immagine non trovata: {local_path}")
    
    return image_paths


def compute_average_image(image_paths, img_size=IMG_SIZE):
    """
    Calcola l'immagine media da una lista di path.
    
    Args:
        image_paths: Lista di path alle immagini
        img_size: Dimensione target (default 128)
    
    Returns:
        Array numpy [img_size, img_size] con l'immagine media normalizzata [0, 1]
    """
    if len(image_paths) == 0:
        return None
    
    avg = np.zeros((img_size, img_size), dtype=np.float32)
    
    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert('L')  # Grayscale
            img = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
            arr = np.array(img, dtype=np.float32) / 255.0
            avg += arr
        except Exception as e:
            print(f"  ⚠️  Errore caricamento {img_path}: {e}")
            continue
    
    if len(image_paths) > 0:
        avg /= len(image_paths)
    
    return avg


def interactive_roi_selection(avg_image, connector_name):
    """
    Mostra l'immagine media e permette di selezionare una ROI interattivamente.
    
    Args:
        avg_image: Array numpy [H, W] con immagine media
        connector_name: Nome del connettore (per il titolo)
    
    Returns:
        ROI come [y1, y2, x1, x2] o None se saltato
    """
    # Variabili per la ROI
    roi_coords = None
    selector = None
    
    def onselect(eclick, erelease):
        """Callback quando viene disegnato un rettangolo."""
        nonlocal roi_coords
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        
        # Normalizza coordinate (assicura x1 < x2, y1 < y2)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # Clamp in [0, IMG_SIZE-1]
        x1 = max(0, min(x1, IMG_SIZE - 1))
        x2 = max(0, min(x2, IMG_SIZE - 1))
        y1 = max(0, min(y1, IMG_SIZE - 1))
        y2 = max(0, min(y2, IMG_SIZE - 1))
        
        roi_coords = [int(y1), int(y2), int(x1), int(x2)]
        print(f"  📦 ROI selezionata: y=[{y1}, {y2}], x=[{x1}, {x2}]")
    
    def on_key_press(event):
        """Callback per i tasti."""
        nonlocal roi_coords, selector
        
        if event.key == 'enter':
            if roi_coords is not None:
                plt.close(fig)
            else:
                print("  ⚠️  Nessuna ROI selezionata. Disegna un rettangolo prima di premere ENTER.")
        
        elif event.key == 'r':
            # Reset
            roi_coords = None
            if selector is not None:
                selector.set_active(False)
                selector.clear()
                selector.set_active(True)
            print("  🔄 ROI resettata. Disegna un nuovo rettangolo.")
        
        elif event.key == 's':
            # Skip
            roi_coords = None
            plt.close(fig)
            print("  ⏭️  Connettore saltato.")
    
    # Crea figura
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(avg_image, cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'PCL ROI Selection - {connector_name}\n'
                 f'Draw rectangle with mouse | ENTER: confirm | R: reset | S: skip',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('X coordinate (0-127)', fontsize=12)
    ax.set_ylabel('Y coordinate (0-127)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Aggiungi RectangleSelector
    selector = RectangleSelector(
        ax,
        onselect,
        useblit=True,
        button=[1],  # Solo tasto sinistro
        minspanx=5, minspany=5,  # Dimensione minima
        spancoords='pixels',
        interactive=True
    )
    
    # Connetti callback per i tasti
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    
    # Mostra istruzioni
    print(f"\n{'='*70}")
    print(f"Annotazione ROI per {connector_name}")
    print(f"{'='*70}")
    print("  📋 Istruzioni:")
    print("     - Disegna un rettangolo con il mouse (click e drag)")
    print("     - ENTER: conferma e passa al prossimo connettore")
    print("     - R: resetta il rettangolo")
    print("     - S: salta questo connettore")
    print()
    
    plt.tight_layout()
    plt.show(block=True)
    
    return roi_coords


def run_pcl_roi_annotation():
    """
    Funzione principale per l'annotazione delle ROI PCL.
    """
    print("="*70)
    print("PCL ROI ANNOTATOR")
    print("="*70)
    print(f"\n📂 CSV dataset: {CSV_PATH}")
    print(f"📂 Base immagini: {DATA_BASE}")
    print(f"💾 Output JSON: {OUTPUT_JSON}")
    
    # Verifica CSV
    if not CSV_PATH.exists():
        print(f"\n❌ ERRORE: CSV non trovato: {CSV_PATH}")
        return
    
    # Carica CSV
    print(f"\n📊 Caricamento dataset...")
    df = pd.read_csv(CSV_PATH)
    print(f"   Totale righe: {len(df)}")
    
    # Estrai connettori unici
    connectors = sorted(df['connector_name'].unique())
    print(f"   Connettori trovati: {len(connectors)}")
    print(f"   Connettori: {', '.join(connectors)}")
    
    # Dizionario per salvare le ROI
    pcl_rois = {}
    
    # Processa ogni connettore
    for idx, connector_name in enumerate(connectors, 1):
        print(f"\n{'='*70}")
        print(f"Connettore {idx}/{len(connectors)}: {connector_name}")
        print(f"{'='*70}")
        
        # Carica immagini
        print(f"📷 Caricamento immagini...")
        image_paths = load_connector_images(connector_name, CSV_PATH, DATA_BASE)
        
        if len(image_paths) == 0:
            print(f"  ⚠️  Nessuna immagine disponibile per {connector_name}. Saltato.")
            continue
        
        print(f"   Immagini caricate: {len(image_paths)}")
        
        # Calcola immagine media
        print(f"🖼️  Calcolo immagine media...")
        avg_image = compute_average_image(image_paths, IMG_SIZE)
        
        if avg_image is None:
            print(f"  ⚠️  Impossibile calcolare immagine media. Saltato.")
            continue
        
        print(f"   ✅ Immagine media calcolata: {avg_image.shape}")
        
        # Selezione ROI interattiva
        roi = interactive_roi_selection(avg_image, connector_name)
        
        if roi is not None:
            pcl_rois[connector_name] = roi
            print(f"  ✅ ROI salvata per {connector_name}: {roi}")
        else:
            print(f"  ⏭️  {connector_name} saltato (nessuna ROI salvata)")
    
    # Salva risultati
    print(f"\n{'='*70}")
    print("SALVATAGGIO RISULTATI")
    print(f"{'='*70}")
    
    if len(pcl_rois) == 0:
        print("⚠️  Nessuna ROI salvata. File JSON non creato.")
        return
    
    # Salva JSON
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(pcl_rois, f, indent=2)
    
    print(f"✅ ROI salvate in: {OUTPUT_JSON}")
    print(f"   Connettori annotati: {len(pcl_rois)}")
    print(f"\n📋 ROI salvate:")
    for conn, roi in sorted(pcl_rois.items()):
        print(f"   {conn}: y=[{roi[0]}, {roi[1]}], x=[{roi[2]}, {roi[3]}]")
    
    print(f"\n✅ Annotazione completata!")


if __name__ == "__main__":
    try:
        run_pcl_roi_annotation()
    except KeyboardInterrupt:
        print("\n\n⚠️  Annotazione interrotta dall'utente.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERRORE: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

