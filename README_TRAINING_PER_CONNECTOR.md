# Training Autoencoder per Connettore - Istruzioni

## Modello Attuale

Stiamo usando un **ConvAE (Convolutional Autoencoder)** con:
- **Encoder**: 3 layer convolutivi (3→16→32→64 canali, stride=2)
- **Decoder**: 3 layer deconvolutivi (64→32→16→3 canali, stride=2)
- **Input/Output**: 128x128x3 (RGB)

## Problema

Attualmente c'è **un solo modello** per tutti i connettori con **un threshold unico**. Questo non è ottimale perché ogni connettore ha caratteristiche diverse.

## Soluzione

Addestrare **9 modelli separati** (uno per connettore: conn1, conn2, ..., conn9), ognuno con il suo threshold specifico.

## File da Creare

Ho creato `step2_autoencoder_per_connector.ipynb` che:
1. Addestra 9 modelli separati
2. Calcola un threshold per ogni connettore
3. Salva modelli come: `models/ae_conv_conn1.pth`, `models/ae_conv_conn2.pth`, ...
4. Salva thresholds come: `models/ae_threshold_conn1.npy`, `models/ae_threshold_conn2.npy`, ...

## Modifiche Necessarie

### 1. Dataset Class

```python
class AEDatasetPerConnector(Dataset):
    def __init__(self, csv_path, connector_name, transform=None):
        df = pd.read_csv(csv_path)
        # Filtra solo OK del connettore specificato
        self.df = df[(df['label'] == 'OK') & (df['connector_name'] == connector_name)].copy()
        self.transform = transform
```

### 2. Training Loop

```python
connectors = [f"conn{i}" for i in range(1, 10)]

for connector_name in connectors:
    # Training
    model = train_autoencoder_per_connector(
        connector_name=connector_name,
        csv_path="data/dataset.csv",
        batch_size=128,
        num_epochs=30,
        device=device
    )
    
    # Calcola threshold
    threshold = calculate_threshold_per_connector(
        model=model,
        connector_name=connector_name,
        csv_path="data/dataset.csv",
        device=device
    )
```

### 3. Salvataggio

- Modelli: `models/ae_conv_{connector_name}.pth`
- Thresholds: `models/ae_threshold_{connector_name}.npy`

## Prossimi Passi

1. Esegui `step2_autoencoder_per_connector.ipynb` su Colab
2. Modifica la GUI (`beko_detection_system.py`) per caricare il modello corretto per ogni connettore
3. Testa con immagini reali

