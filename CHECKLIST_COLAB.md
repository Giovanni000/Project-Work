# Checklist Pre-Esecuzione su Google Colab

## ✅ File Necessari Prima di Eseguire il Notebook

### 1. **Repository GitHub** (Opzionale ma Consigliato)
- ✅ Il notebook clona automaticamente il repository se configurato
- ⚠️ **IMPORTANTE**: Modifica `GITHUB_REPO` nella cella di setup se necessario
- Se non usi GitHub, assicurati che il notebook sia caricato direttamente su Colab

### 2. **Google Drive - Struttura Cartelle**
Le immagini devono essere su Google Drive nella seguente struttura:

```
/content/drive/MyDrive/Project Work/
├── Data/
│   └── connectors/
│       ├── conn1/
│       │   ├── 20251106110559_TOP.png
│       │   ├── 20251106110633_TOP.png
│       │   └── ... (tutte le immagini OK/KO/OCCLUSION)
│       ├── conn2/
│       ├── conn3/
│       ├── ...
│       └── conn9/
└── (opzionale: altri file del progetto)
```

**Verifica:**
- ✅ Le immagini devono essere accessibili da Colab
- ✅ I path devono corrispondere esattamente a: `/content/drive/MyDrive/Project Work/Data/connectors/{connector_name}/{filename}`

### 3. **File CSV** (Opzionale - Il Notebook Li Crea Automaticamente)

Il notebook può creare `data/dataset.csv` automaticamente se hai `features_labeled.csv`.

**Opzione A: Usa `features_labeled.csv` (Consigliato)**
- ✅ Carica `features_labeled.csv` nella root del repository su Colab
- ✅ Il notebook lo trova automaticamente e crea `data/dataset.csv`
- ✅ Formato richiesto: colonne `filename`, `label`, `connector_name`

**Opzione B: Usa `data/dataset.csv` già pronto**
- ✅ Carica `data/dataset.csv` nella cartella `data/` del repository
- ✅ Formato richiesto: colonne `image_path`, `label`, `connector_name`
- ✅ I path devono puntare a Google Drive: `/content/drive/MyDrive/Project Work/Data/connectors/...`

### 4. **File NON Necessari (Creati Automaticamente)**
- ❌ `models/` - Viene creata automaticamente
- ❌ `spatial_mask_*.npy` - Vengono calcolati durante il training
- ❌ `efficientad_student_*.pth` - Vengono salvati durante il training
- ❌ `efficientad_threshold_*.npy` - Vengono calcolati dopo il training

---

## 🚀 Procedura di Esecuzione su Colab

### Step 1: Preparazione
1. **Apri Google Colab**
2. **Carica il notebook** `step2_efficientad_per_connector.ipynb`
   - Opzione A: Carica direttamente il file .ipynb
   - Opzione B: Clona repository GitHub (modifica `GITHUB_REPO` se necessario)

### Step 2: Verifica Setup
1. **Esegui la cella di Setup** (Cell 2)
   - Monta Google Drive quando richiesto
   - Verifica che il repository sia clonato/caricato correttamente
   - Verifica che `DATA_ROOT` punti alla cartella corretta

2. **Verifica Dataset CSV** (Cell 4)
   - Se `data/dataset.csv` esiste → ✅ OK
   - Se non esiste → Il notebook cerca `features_labeled.csv` e lo crea automaticamente
   - ⚠️ Se entrambi mancano → ERRORE (devi caricare almeno uno dei due)

### Step 3: Verifica Immagini
Dopo aver montato Drive, verifica che le immagini siano accessibili:

```python
# Cella di test (opzionale)
from pathlib import Path
test_path = Path("/content/drive/MyDrive/Project Work/Data/connectors/conn1")
if test_path.exists():
    images = list(test_path.glob("*.png"))
    print(f"✅ Trovate {len(images)} immagini in conn1")
else:
    print("❌ Cartella conn1 non trovata su Drive!")
```

### Step 4: Esegui Training
1. **Esegui tutte le celle in ordine**
2. Il notebook:
   - Calcola spatial mask per ogni connettore (se non esiste)
   - Addestra 9 modelli EfficientAD-M (uno per connettore)
   - Calcola threshold per ogni connettore
   - Salva tutto in `models/`

### Step 5: Visualizzazione (Opzionale)
Dopo il training, puoi eseguire le funzioni di visualizzazione:
- `visualize_spatial_mask()`
- `visualize_average_anomaly_map_for_connector()`
- `visualize_example_ok_ko_heatmaps()`

---

## ⚠️ Problemi Comuni e Soluzioni

### Problema 1: "FileNotFoundError: features_labeled.csv non trovato"
**Soluzione:**
- Carica `features_labeled.csv` nella root del repository su Colab
- Oppure carica direttamente `data/dataset.csv` già pronto

### Problema 2: "Nessuna immagine trovata"
**Soluzione:**
- Verifica che Google Drive sia montato correttamente
- Verifica che i path siano esatti: `/content/drive/MyDrive/Project Work/Data/connectors/...`
- Controlla che le immagini siano nella struttura corretta

### Problema 3: "SSL Certificate Error" (caricamento ResNet18)
**Soluzione:**
- Il codice gestisce automaticamente questo errore
- Se persiste, il Teacher userà ResNet18 senza pre-trained weights (performance ridotta)

### Problema 4: Training troppo lento
**Soluzione:**
- Le immagini su Drive sono lente da caricare
- Considera di copiare le immagini in locale su Colab prima del training:
  ```python
  # Cella opzionale: copia immagini in locale
  import shutil
  LOCAL_DATA = "/content/local_data"
  DRIVE_DATA = "/content/drive/MyDrive/Project Work/Data/connectors"
  shutil.copytree(DRIVE_DATA, LOCAL_DATA, dirs_exist_ok=True)
  # Poi modifica i path nel CSV o nel codice
  ```

---

## 📋 Checklist Rapida Pre-Esecuzione

Prima di eseguire su Colab, verifica:

- [ ] Google Drive contiene le immagini in `/content/drive/MyDrive/Project Work/Data/connectors/`
- [ ] Hai `features_labeled.csv` OPPURE `data/dataset.csv` pronto
- [ ] Il notebook è caricato su Colab (o repository GitHub configurato)
- [ ] GPU abilitata su Colab (Runtime → Change runtime type → GPU)

---

## 🎯 Dopo l'Esecuzione

Il notebook creerà automaticamente:

```
models/
├── spatial_mask_conn1.npy
├── spatial_mask_conn2.npy
├── ...
├── spatial_mask_conn9.npy
├── efficientad_student_conn1.pth
├── efficientad_student_conn2.pth
├── ...
├── efficientad_student_conn9.pth
├── efficientad_threshold_conn1.npy
├── efficientad_threshold_conn2.npy
├── ...
└── efficientad_threshold_conn9.npy
```

**Per usare i modelli nel sistema GUI:**
- Scarica la cartella `models/` da Colab
- Copiala in `Training/models/` nel progetto locale
- I modelli sono compatibili con `beko_detection_system.py`

---

## 💡 Suggerimenti

1. **Prima esecuzione**: Testa con un solo connettore (modifica il loop per processare solo `conn1`)
2. **Monitoraggio**: Controlla i progress bar durante il training
3. **Salvataggio**: Colab può disconnettersi - salva i modelli man mano
4. **Performance**: Usa GPU (T4) per training più veloce

