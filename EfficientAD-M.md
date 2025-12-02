Perfetto, ecco **il file `.md` riscritto correttamente**, aggiornato per il tuo caso reale:

* **9 connettori completamente diversi tra loro**
* **KO disponibili solo per alcuni**
* **1 modello EfficientAD-M per ogni connettore**
* **soglie gestite automaticamente anche quando mancano i KO**

La struttura è pulita, ordinata, pronta da incollare su Cursor / GitHub.
Tutto quello che rimane uguale viene mantenuto, tutto ciò che cambia rispetto al file originale è stato corretto in modo coerente.
Dove uso parti prese dal file originale, le cito. 

---

# EfficientAD-M — Pipeline per Connettori Diversi (1 Modello per Connettore)

Questo documento definisce la pipeline di anomaly detection basata su **EfficientAD-M**, adattata al caso in cui esistono **9 connettori completamente diversi** per forma, colore e dimensione.

L’obiettivo è:

* addestrare **un modello per ciascun connettore**
* usare solo immagini **OK** per il training
* gestire soglie differenti a seconda della presenza o meno di KO
* produrre anomaly score e heatmap per ogni connettore

---

## 📁 1. Struttura del Dataset

Per ogni connettore `conn1 … conn9`:

```
data/
  conn1/
    train/ok/
    test/ok/
    test/ko/   # può essere vuota o mancante
  conn2/
    train/ok/
    test/ok/
    test/ko/
  ...
  conn9/
    train/ok/
    test/ok/
    test/ko/
```

### Regole

* `train/ok/` → SOLO immagini OK del **singolo** connettore
* `test/ok/` → OK di test
* `test/ko/` → immagini difettose, mancanti, occluse *se presenti*
* Se un connettore NON ha KO → la cartella può essere vuota o omessa

---

## 🧱 2. Installazione dipendenze

```bash
pip install torch torchvision
```

---

## 🎨 3. Trasformazioni immagine

*(coerenti con il file originale)* 

```python
import torchvision.transforms as T

IMG_SIZE = 256

train_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.Grayscale(num_output_channels=3),
    T.ToTensor(),
    T.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

test_transform = train_transform
```

---

## 📦 4. Dataset e DataLoader

*(basato sul file originale)* 

```python
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset

class ConnectorDataset(Dataset):
    def __init__(self, folder, label, transform):
        self.folder = Path(folder)
        if self.folder.exists():
            self.files = sorted(self.folder.glob("*"))
        else:
            self.files = []
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        img = Image.open(p).convert("RGB")
        img = self.transform(img)
        return img, self.label, str(p)
```

---

## 🧠 5. Modelli Teacher & Student

*(identico al file originale)* 

### Teacher — ResNet18 pre-trained

```python
import torch
import torch.nn as nn
import torchvision.models as models

class Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(base.children())[:-2])

    def forward(self, x):
        return self.encoder(x)
```

### Student — ResNet18 non pre-trained

```python
class Student(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=None)
        self.encoder = nn.Sequential(*list(base.children())[:-2])

    def forward(self, x):
        return self.encoder(x)
```

---

## 🟦 6. Training — Solo sugli OK

*(coerente con il file originale)* 

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(train_loader):
    teacher = Teacher().to(device)
    student = Student().to(device)

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    EPOCHS = 20
    for e in range(EPOCHS):
        student.train()
        losses = []
        for imgs, _, _ in train_loader:
            imgs = imgs.to(device)
            with torch.no_grad():
                t_feat = teacher(imgs)
            s_feat = student(imgs)

            loss = criterion(s_feat, t_feat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        print(f"Epoch {e+1}/{EPOCHS} - loss {sum(losses)/len(losses):.6f}")

    return student
```

---

## 🔍 7. Anomaly Scoring

*(derivato dal file originale)* 

```python
import numpy as np
from tqdm import tqdm

def compute_scores(teacher, student, dataloader):
    teacher.eval(); student.eval()
    all_scores, all_labels, all_paths = [], [], []

    with torch.no_grad():
        for imgs, labels, paths in tqdm(dataloader):
            imgs = imgs.to(device)
            t = teacher(imgs)
            s = student(imgs)
            diff = (t - s)**2
            amap = diff.mean(dim=1)        # B, Hf, Wf
            scores = amap.flatten(1).max(1)[0].cpu().numpy()

            all_scores.extend(scores)
            all_labels.extend(labels.numpy())
            all_paths.extend(paths)

    return np.array(all_scores), np.array(all_labels), all_paths
```

---

## 🎯 8. Gestione Soglie

### 8.1 — Se il connettore **ha KO**

Usa metriche supervisionate:

```python
from sklearn.metrics import roc_auc_score

auc = roc_auc_score(labels, scores)
threshold = np.percentile(scores[labels==0], 95)
```

### 8.2 — Se il connettore **NON ha KO**

Usa soglia completamente unsupervised:

```python
threshold = np.percentile(scores, 95)
```

Oppure scegli un percentile comune derivato dagli altri connettori.

---

## 🔥 9. Heatmap

*(identico alla versione del file)* 

```python
import cv2
import os

os.makedirs("results/heatmaps", exist_ok=True)

def save_heatmap(img, amap, out_path):
    img = img.cpu().numpy().transpose(1,2,0)
    img = (img*0.5+0.5)*255
    img = img.astype(np.uint8)

    a = amap.cpu().numpy()
    a = cv2.resize(a, (img.shape[1], img.shape[0]))
    a = (a - a.min())/(a.max()-a.min()+1e-8)
    a = (a*255).astype(np.uint8)

    heat = cv2.applyColorMap(a, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heat, 0.4, 0)
    cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
```

---

## 🔁 10. Training Loop per Tutti i Connettori

```python
connector_ids = [1,2,3,4,5,6,7,8,9]

for cid in connector_ids:
    base = f"data/conn{cid}"

    train_ok = ConnectorDataset(f"{base}/train/ok", 0, train_transform)
    test_ok  = ConnectorDataset(f"{base}/test/ok", 0, test_transform)
    test_ko  = ConnectorDataset(f"{base}/test/ko", 1, test_transform)

    train_loader = DataLoader(train_ok, batch_size=32, shuffle=True)
    test_loader  = DataLoader(ConcatDataset([test_ok, test_ko]),
                              batch_size=32, shuffle=False)

    print(f"\n=== Training connettore {cid} ===")
    student = train_model(train_loader)

    torch.save(student.state_dict(), f"models/efficientad_conn{cid}.pth")

    # Valutazione
    teacher = Teacher().to(device)
    scores, labels, paths = compute_scores(teacher, student, test_loader)
```

---

## 🧩 11. Note Finali

* EfficientAD-M impara la **normalità** del singolo connettore
* Le anomalie sono riconosciute come deviazioni feature teacher → student
* Non serve alcun KO per addestrare
* I KO servono **solo per calibrare la soglia**, e solo quando disponibili

---

## Se vuoi:

Ti preparo anche:

* un **notebook Jupyter completo**
* oppure una **versione “Cursor-ready”** con file creati automaticamente:

  * `train_connector.py`
  * `dataset.py`
  * `efficientad.py`
  * `config.yaml`

Dimmi quale vuoi e te lo genero subito.
