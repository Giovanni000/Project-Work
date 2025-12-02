

**PROMPT PER CURSOR (COPIA DA QUI IN GIÙ)**

Voglio che tu imposti da zero una pipeline di **anomaly detection con PatchCore** per il mio dataset di connettori (OK/KO), usando **PyTorch**.

⚙️ **Contesto generale (molto importante)**

* Ho un dataset di **crop di immagini** di connettori su PCB, con:

  * tantissimi **OK**
  * pochissimi **KO** (circa 25 su ~2500 crop totali, ordine di grandezza)
* I KO principali sono:

  * **mancanza del connettore** (si vede più board bianca sotto)
  * **occlusioni** con cavi davanti
* Le anomalie sono **principalmente strutturali/geometriche** (presenza/assenza del connettore, cavi, bordi), non solo cromatiche.
* Puoi usare la struttura di cartelle che trovi nel repo per capire dove sono le immagini; se necessario, crea una piccola sezione di config all’inizio con i path modificabili.

🎯 **Obiettivo principale**

Implementa una pipeline PatchCore che faccia queste cose:

1. Usa **solo immagini OK** per costruire la memoria di PatchCore (training set).
2. Valuta su un set di immagini di test che contiene:

   * OK
   * KO (mancanza connettore, occlusioni, ecc.)
3. Produci:

   * uno **score di anomalia per immagine**
   * una **heatmap** che mostri *dove* è l’anomalia sul connettore
   * metriche tipo **ROC-AUC**, **PR-AUC**, istogrammi degli score OK vs KO
4. Permetti facilmente di provare sia **grayscale** che **RGB** con un semplice flag.

📁 **Struttura codice che voglio**

Crea un file principale, ad esempio:

* `patchcore_experiments.py` **oppure** una notebook `patchcore_experiments.ipynb`

  * scegli tu, ma organizza il codice in modo pulito e modulare.
  * se serve, crea anche un file di supporto tipo `patchcore.py` con le classi principali.

Cerca di **non rompere** il codice esistente: aggiungi file nuovi, riusa funzioni/utility già presenti nel repo se hanno senso (es. loader di immagini, trasformazioni, ecc.).

---

### 1️⃣ Configurazione iniziale

All’inizio del file principale, crea una sezione di configurazione chiara, tipo:

* `DATA_ROOT` : path di base del dataset
* `TRAIN_OK_DIR` : cartella con le immagini OK di training
* `TEST_OK_DIR`  : cartella con le immagini OK di test
* `TEST_KO_DIR`  : cartella con le immagini KO di test
* `IMG_SIZE` (es. 128 o 256)
* `USE_GRAYSCALE = True` (default True, ma se metto False usa RGB)
* `BATCH_SIZE`, `NUM_WORKERS`
* `BACKBONE = "resnet18"` (va benissimo resnet18 o simile)
* `PATCH_STRIDE`, `PATCH_LAYER` ecc. (parametri PatchCore)
* `DEVICE` (usa automaticamente `"cuda"` se disponibile)

Se non riesci a inferire la struttura esatta delle cartelle, fai in modo che tutto sia facilmente configurabile da queste variabili.

---

### 2️⃣ Data pipeline (Dataset & DataLoader)

Implementa un sistema dati così:

1. **Dataset personalizzato** (es. `ConnectorDataset`):

   * riceve:

     * lista di path immagini
     * label (0 = OK, 1 = KO)
     * flag `grayscale=True/False`
   * trasformazioni:

     * resize a `IMG_SIZE x IMG_SIZE`
     * se `grayscale=True`:

       * converte a singolo canale ma in PyTorch rimane shape `[1, H, W]`
     * normalizzazione standard (tipo mean/std = 0.5/0.5 per semplicità, o quelle di ImageNet se usi RGB e backbone pre-addestrato)

2. **DataLoader**:

   * `train_loader`: solo immagini **OK** per costruire la memoria di PatchCore
   * `test_loader`: immagini OK + KO, con label, per valutare.

Organizza bene il codice per non duplicare logica tra train e test.

---

### 3️⃣ Implementazione PatchCore

Implementa una versione funzionale di PatchCore, anche semplificata, ma completa. Struttura suggerita:

1. **Feature extractor (backbone)**

   * Usa una rete pre-addestrata tipo `torchvision.models.resnet18(pretrained=True)`
   * Rimuovi l’ultimo fc e usa una **feature map intermedia** (ad esempio l’output di un layer conv).
   * Congela i pesi (no training, solo feature extraction).

2. **Estrazione feature patch-wise**

   * Per ogni immagine:

     * passa nell’encoder, ottieni feature map `[C, Hf, Wf]`
     * “appiattisci” in una lista di patch feature `[N_patches, C]` con sliding/stride
   * Salva queste patch feature per tutte le immagini **OK di train**.

3. **Memoria compatta (coreset)**

   * Possibile strategia semplice:

     * concatena tutte le patch feature degli OK
     * se sono troppe, applica una **selezione random** o una coreset selection (puoi implementare una coreset greedy semplice o usare un sottocampionamento casuale, specifica nei commenti cosa stai facendo)
   * Questa memoria sarà la "PatchCore memory".

4. **Anomaly scoring**

   * Per un’immagine di test:

     * estrai le patch feature come sopra
     * per ogni patch, trova il **nearest neighbor** nella memoria (puoi usare `faiss` se vuoi, o `sklearn.NearestNeighbors`, o una semplice distanza euclidea con batching)
     * ottieni distanze patch-wise
   * Definisci:

     * score di patch = distanza alla memoria
     * score di immagine = max (o una combinazione tipo top-k mean) delle distanze delle patch

5. **Heatmap**

   * Rimappa gli score delle patch su una griglia `[Hf, Wf]`
   * Upsample alla dimensione originale dell’immagine (bilinear)
   * Normalizza 0–1
   * Salva:

     * heatmap come immagine
     * overlay con l’immagine originale (usa una colormap tipo jet su OpenCV o matplotlib).

Tutto questo incapsulato in una classe tipo `PatchCoreModel` con metodi:

* `fit(train_loader)`
* `predict_scores(test_loader)` → ritorna:

  * lista di image-level anomaly scores
  * opzionale: heatmaps per alcune immagini campione
* `save_results(...)` per salvare CSV e immagini di debug.

---

### 4️⃣ Training / Fitting PatchCore

Nella funzione `main()` (o nell’ultima cella del notebook):

1. Carica i dati:

   * costruisci train_loader con soli OK
   * costruisci test_loader con OK/KO

2. Istanzia `PatchCoreModel` con i parametri di config.

3. Chiama `fit(train_loader)`:

   * estrazione feature
   * costruzione memoria

4. Poi eval:

   * `scores, labels = model.evaluate(test_loader)`
     (dove `labels` sono i 0/1 veri)
   * Calcola:

     * ROC-AUC
     * PR-AUC
     * istogrammi score OK vs KO
   * Stampa risultati in modo pulito.

5. Salva un file CSV tipo `patchcore_results.csv` con:

   * path immagine
   * label reale
   * anomaly score
   * eventuale flag “predicted anomaly” usando una soglia (es. Otsu, o percentile).

---

### 5️⃣ Visualizzazione risultati (heatmap e debug)

Implementa una funzione tipo `visualize_examples(...)` che:

* seleziona qualche immagine:

  * alcuni OK
  * alcuni KO
* per ognuna:

  * genera la heatmap di anomalia
  * crea e salva:

    * `original.png`
    * `heatmap.png`
    * `overlay.png` (originale + heatmap semi-trasparente)

Salvale in una cartella tipo `results/patchcore_visuals/`.

---

### 6️⃣ Modalità grayscale vs RGB

Implementa una modalità comoda per switchare:

* Se `USE_GRAYSCALE = True`:

  * converte l’immagine a singolo canale
  * ma nel backbone:

    * o replichi il canale a 3 per usare ResNet pre-addestrata (`x.repeat(3,1,1)`)
    * o usi un primo layer custom adattato a 1 canale
* Se `USE_GRAYSCALE = False`:

  * usa RGB normale con le mean/std di ImageNet.

Il default che voglio è **grayscale**, con opzione facile per testare RGB.

---

### 7️⃣ Buone pratiche

* Usa **tipi annotati** e docstring sintetiche per le funzioni principali.
* Commenta i passaggi importanti (specialmente quelli di PatchCore: coreset, NN search, scoring).
* Gestisci `device` (`cuda`/`cpu`) in modo pulito e centralizzato.
* Non fare codice “monolitico” unico enorme: suddividi in funzioni/metodi leggibili.
* Stampa qualche log durante `fit` ed eval (numero patch, dimensione memoria, ecc.).

---

### 8️⃣ Output finale atteso

Quando eseguo il file/notebook, voglio ottenere:

1. Stampa in console:

   * numero immagini train/test
   * dimensioni memoria patch
   * ROC-AUC e PR-AUC
2. File:

   * `results/patchcore_results.csv`
   * `results/patchcore_hist_scores.png` (istogrammi score OK vs KO)
   * `results/patchcore_roc_curve.png`
   * `results/patchcore_pr_curve.png`
   * `results/patchcore_visuals/` con alcune heatmap/overlay significative

Organizza il tutto in modo chiaro, pronto per sperimentare vari parametri (es. IMG_SIZE, backbone, grayscale vs RGB).

---

Usa tutta la conoscenza che hai della struttura attuale del mio repo per integrare questa pipeline senza rompere nulla, riusando se possibile utilità già esistenti (es. per caricare immagini, logging, ecc.).
