# BEKO DETECTION SYSTEM

Interfaccia grafica per analisi PCB connector con modelli deep learning.

## 🚀 Caratteristiche

- **Interfaccia grafica moderna** con drag & drop
- **Preprocessing automatico**: allineamento e normalizzazione illuminazione
- **Estrazione automatica** dei 9 connettori
- **Classificazione** con modelli deep learning (OK/KO/OCCLUSION)
- **Visualizzazione grafica** completa dei risultati

## 📋 Requisiti

### Dipendenze Python

```bash
pip install tkinterdnd2 torch torchvision opencv-python pillow matplotlib numpy
```

**Nota:** `tkinter` è incluso in Python, ma su alcuni sistemi potrebbe essere necessario installarlo separatamente:
- **Ubuntu/Debian**: `sudo apt-get install python3-tk`
- **macOS**: tkinter è incluso con Python
- **Windows**: tkinter è incluso con Python

### File Necessari

Il programma cerca automaticamente:

1. **Modelli addestrati** in `Training/models/`:
   - `occlusion_cnn.pth` (classificatore OCCLUSION)
   - `ae_conv.pth` (autoencoder)
   - `ae_threshold.npy` (threshold)

2. **Configurazione ROI** in `Codice/roi_config.json`

## 🎯 Utilizzo

### Avvio

```bash
python3 Training/beko_detection_system.py
```

### Procedura

1. **Carica immagine**:
   - Trascina un'immagine PCB nell'area di upload
   - Oppure clicca per selezionare un file

2. **Clicca "ANALIZZA IMMAGINE"**

3. **Attendi l'elaborazione**:
   - Allineamento immagine
   - Estrazione connettori
   - Classificazione (9 connettori)

4. **Visualizza risultati**:
   - Immagine principale con bounding box colorati
   - Grid 3x3 con tutti i connettori
   - Statistiche nella sidebar

## 🎨 Colori

- **Verde (OK)**: Connettore corretto
- **Rosso (KO)**: Connettore con anomalia
- **Arancione (OCCLUSION)**: Connettore occluso

## ⚙️ Configurazione

Il programma cerca automaticamente i file nella struttura del progetto:

```
Project Work/
├── Training/
│   ├── models/
│   │   ├── occlusion_cnn.pth
│   │   ├── ae_conv.pth
│   │   └── ae_threshold.npy
│   └── beko_detection_system.py
└── Codice/
    └── roi_config.json
```

Se i file sono in posizioni diverse, modifica i path nel codice:
- `self.models_dir`
- `self.roi_config_path`

## 🔧 Risoluzione Problemi

### "Modello non trovato"
- Assicurati di aver eseguito step1 e step2 per addestrare i modelli
- Verifica che i file siano in `Training/models/`

### "ROI config non trovato"
- Verifica che `Codice/roi_config.json` esista
- Controlla il path nel codice

### "tkinterdnd2 non installato"
- Esegui: `pip install tkinterdnd2`

### L'interfaccia non si apre
- Verifica che tkinter sia installato
- Su Linux: `sudo apt-get install python3-tk`

## 📊 Performance

- **CPU**: ~1-2 secondi per analizzare 9 connettori
- **GPU**: ~0.5-1 secondo (se disponibile)

L'inferenza è veloce anche su CPU!

## 🎯 Note

- Il programma assume che l'immagine sia già allineata (non richiede riferimento)
- Se l'immagine non è allineata, modifica `align_image()` per usare un riferimento
- Il preprocessing include normalizzazione illuminazione automatica

