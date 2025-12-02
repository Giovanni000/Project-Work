# Spiegazione: Come Funziona la Classificazione OK vs KO

## Panoramica del Sistema

Il sistema di classificazione funziona in **2 step**:

### STEP 1: Classificatore OCCLUSION vs VISIBLE

- **Input**: Immagine del connettore (128x128, RGB)
- **Output**: 
  - `OCCLUSION` (classe 0) → Il connettore è occluso/parzialmente occluso
  - `VISIBLE` (classe 1) → Il connettore è visibile (può essere OK o KO)

Se il risultato è `OCCLUSION`, il sistema ritorna immediatamente `"OCCLUSION"` e non procede oltre.

### STEP 2: Autoencoder per Anomaly Detection (solo se VISIBLE)

Se il connettore è `VISIBLE`, il sistema procede con l'**autoencoder** per distinguere tra `OK` e `KO`.

#### Come Funziona l'Autoencoder

1. **Training**:
   - L'autoencoder è stato addestrato **SOLO su immagini OK**
   - Ha imparato a ricostruire correttamente le immagini di connettori OK
   - Non ha mai visto immagini KO durante il training

2. **Inference**:
   - L'autoencoder riceve l'immagine del connettore
   - Cerca di ricostruirla
   - Calcola l'**errore di ricostruzione** (MSE - Mean Squared Error) tra l'immagine originale e quella ricostruita

3. **Decisione**:
   - Se l'errore è **basso** → Il connettore è simile a quelli OK visti in training → **OK**
   - Se l'errore è **alto** → Il connettore è diverso/anomalo → **KO**

#### Calcolo del Threshold

Il **threshold** viene calcolato durante il training:

```
threshold = mu + 3*sigma
```

Dove:
- `mu` = media degli errori di ricostruzione su tutte le immagini OK del dataset di training
- `sigma` = deviazione standard degli errori di ricostruzione

**Regola di decisione**:
- `errore > threshold` → **KO** (anomalia)
- `errore ≤ threshold` → **OK** (normale)

## Problema: KO Classificati come OK

Se alcuni connettori KO vengono classificati come OK, possibili cause:

### 1. Threshold Troppo Alto

Se il threshold è troppo alto, molti KO avranno un errore di ricostruzione inferiore al threshold e verranno classificati erroneamente come OK.

**Soluzione**: Ridurre il threshold (es. usare `mu + 2*sigma` invece di `mu + 3*sigma`)

### 2. Autoencoder Non Ha Imparato Bene

Se l'autoencoder non ha imparato bene a ricostruire le immagini OK, potrebbe avere errori alti anche su OK, rendendo difficile distinguere OK da KO.

**Soluzione**: 
- Aumentare il numero di epoche di training
- Aumentare la capacità del modello (più layer/neuroni)
- Verificare che il dataset di training contenga solo immagini OK ben preprocessate

### 3. Preprocessing Non Identico

Se il preprocessing delle immagini durante l'inference è diverso da quello del training, l'autoencoder potrebbe non funzionare correttamente.

**Verifica**:
- Nel training: immagini caricate come RGB, resize a 128x128, convertite a tensor [0,1]
- Nell'inference: immagini in grayscale, convertite a RGB, resize a 128x128, convertite a tensor [0,1]

**Nota**: Il preprocessing dovrebbe essere identico. Se le immagini di training erano in RGB e quelle di inference sono in grayscale convertite a RGB, potrebbe esserci una differenza.

### 4. Dataset di Training Non Rappresentativo

Se il dataset di training contiene solo alcuni tipi di connettori OK, l'autoencoder potrebbe non generalizzare bene su altri connettori OK o potrebbe non distinguere bene i KO.

## Come Verificare il Problema

1. **Controlla il threshold**: Viene stampato nella console all'avvio del programma
2. **Controlla gli errori**: Nella visualizzazione, ogni connettore mostra il suo errore di ricostruzione
3. **Confronta errori**: 
   - Se connettori OK hanno errori simili a KO → threshold troppo alto o autoencoder non funziona bene
   - Se connettori KO hanno errori molto bassi → autoencoder non ha imparato bene o preprocessing diverso

## Suggerimenti per Migliorare

1. **Ridurre il threshold**: Modifica `step2_autoencoder.ipynb` per usare `mu + 2*sigma` invece di `mu + 3*sigma`
2. **Aggiungere più dati OK al training**: Più varietà di connettori OK migliora la capacità dell'autoencoder
3. **Verificare preprocessing**: Assicurati che il preprocessing sia identico tra training e inference
4. **Aumentare capacità modello**: Più layer o più neuroni possono aiutare l'autoencoder a imparare meglio

