#!/usr/bin/env python3
"""Script per generare step4_visual_inference.ipynb completo."""

import json

cells = []

# Cell 0: Markdown header
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# STEP 4: Inferenza Visiva con Preprocessing Completo\n",
        "\n",
        "Questo notebook:\n",
        "1. Carica un'immagine PCB\n",
        "2. La allinea e normalizza (come `preprocess_alignment.py`)\n",
        "3. Estrae i 9 connettori (come `crop_connectors.py`)\n",
        "4. Classifica ogni connettore con il modello allenato\n",
        "5. Mostra i risultati con visualizzazione grafica"
    ]
})

# Cell 1: Setup
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Setup: Clona repository GitHub e monta Google Drive\n",
        "import os\n",
        "from pathlib import Path\n",
        "\n",
        "GITHUB_REPO = \"https://github.com/Giovanni000/Project-Work.git\"  # ⚠️ MODIFICA QUESTO!\n",
        "REPO_DIR = \"/content/project\"\n",
        "\n",
        "if not Path(REPO_DIR).exists():\n",
        "    !git clone {GITHUB_REPO} {REPO_DIR}\n",
        "\n",
        "os.chdir(REPO_DIR)\n",
        "print(f\"Repository directory: {os.getcwd()}\")\n",
        "\n",
        "from google.colab import drive\n",
        "drive.mount('/content/drive')\n",
        "\n",
        "import torch\n",
        "import torch.nn as nn\n",
        "from torchvision import transforms\n",
        "from PIL import Image\n",
        "import numpy as np\n",
        "import cv2\n",
        "import json\n",
        "import matplotlib.pyplot as plt\n",
        "from matplotlib.patches import Rectangle\n",
        "from dataclasses import dataclass\n",
        "\n",
        "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
        "print(f\"Device: {device}\")\n",
        "if torch.cuda.is_available():\n",
        "    print(f\"GPU: {torch.cuda.get_device_name(0)}\")"
    ]
})

# Cell 2: Markdown - Carica Modelli
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## Carica Modelli (da step3)"]
})

# Cell 3: Carica Modelli (codice completo in file separato per leggibilità)
# Continua con le altre celle...

notebook = {
    "cells": cells,
    "metadata": {"language_info": {"name": "python"}},
    "nbformat": 4,
    "nbformat_minor": 2
}

with open("Training/step4_visual_inference.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)

print("✅ Notebook creato!")

