# ERCP Image Classification — CNN + XGBoost Hybrid Pipeline

Classificação automática de imagens de CPRE (Colangiopancreatografia Retrógrada Endoscópica) em 4 classes:
**Biliary_Leaks · Lithiasis · Normal · Stricture**

Pipeline híbrido: backbone CNN pré-treinado → extração de features → XGBoost com GridSearch → ensemble de 4 modelos.

**Resultado principal:** F1 macro = **0.7934** (threshold Stricture ajustado: recall Stricture = 90.5%)  
**Baseline do paper:** F1 macro = 0.738

---

## Estrutura do Repositório

```
.
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 1_split_dataset.ipynb         # PASSO 1 — split patient-aware
│   ├── 2_ercp_hybrid_pipeline.ipynb  # PASSO 2 — pipeline principal (resultados finais)
│   ├── models/                       # Modelos usados pelo pipeline
│   │   ├── densenet.pth
│   │   ├── resnet50.pth
│   │   ├── efficientnet_b7.pth
│   │   ├── mobilenet_v2.pth
│   │   ├── deitiii.pth
│   │   └── resnet_xgb.pkl
│   ├── preds/                        # Previsões e figuras geradas
│   ├── imagens/                      # Figuras EDA e comparações
│   ├── DEITIII.ipynb             # DeiT-III Small
│   ├── DENSENET.ipynb            # DenseNet121 com CLAHE
│   ├── EFICIENTNET.ipynb         # EfficientNet-B7 com CLAHE
│   ├── MOBILENET.ipynb           # MobileNet-V2 com CLAHE
│   └── RESNET.ipynb              # ResNet50 com CLAHE
└── dataset/                          # Dataset MIQR-CC (não incluído — ver abaixo)
    ├── train/
    │   ├── Biliary_Leaks/
    │   ├── Lithiasis/
    │   ├── Normal/
    │   └── Stricture/
    ├── val/
    └── test/
```

---

## Dataset

O dataset MIQR-CC não é distribuído com este repositório.  
Após obter as imagens, o split patient-aware é gerado pelo notebook `1_split_dataset.ipynb`.

**Dimensões do split:**
| Split | Total | Biliary_Leaks | Lithiasis | Normal | Stricture |
|-------|-------|---------------|-----------|--------|-----------|
| train | 1067  | 110 | 505 | 197 | 255 |
| val   | 234   | 24  | 98  | 59  | 53  |
| test  | 267   | 17  | 123 | 43  | 84  |

---

## Setup do Ambiente

```bash
# Criar e activar ambiente virtual
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Instalar dependências
pip install -r requirements.txt

# GPU (CUDA 12.1):
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

> **Nota:** O projeto foi desenvolvido com Python 3.11 e CUDA 12.1. CPU é suportado mas o treino e extração de features serão mais lentos (~5–10×).

---

## Ordem de Execução

### Passo 1 — Split patient-aware

Abre e corre `notebooks/1_split_dataset.ipynb`.

Este notebook lê o dataset original e cria o split em `dataset/` com isolamento de pacientes entre train/val/test.

**Output:** pastas `dataset/train/`, `dataset/val/`, `dataset/test/`

---

### Passo 2 (Opcional) — Re-treinar os backbones

Se quiseres re-treinar os modelos do zero, corre os notebooks pela seguinte ordem:

```
DEITIII.ipynb
DENSENET.ipynb
RESNET.ipynb
EFICIENTNET.ipynb
MOBILENET.ipynb
```

Cada notebook guarda o modelo em `notebooks/models/`.

**Configurações de treino (iguais para todos os modelos):**
- Optimizador: Adam (lr=1e-4)
- Loss: FocalLoss (gamma=2)
- Scheduler: CosineAnnealingLR
- Early stopping: patience=10
- Epochs máx: 30
- Pré-processamento: Resize(512×512) → CLAHE → Augmentation → NormalizeIntensity

---

### Passo 3 — Pipeline Híbrido (resultados principais)

Abre `notebooks/2_ercp_hybrid_pipeline.ipynb`.

O notebook está organizado em secções executáveis de forma independente:

| Secção | Descrição|
|--------|-----------|
| CONFIG | Define `MODEL_TYPE` e `MODEL_PTH`|
| 1 | Carregar listas de imagens|
| 2 | Transforms (sem augmentation)|
| 3 | Carregar modelo e cortar cabeça|
| 4 | Extrair features + baseline CNN|
| 5 | Treinar XGBoost (GridSearch)|
| 6 | Avaliar no test set + matriz de confusão|
| 7 | **ENSEMBLE** (todos os modelos)|
| 8 | Grad-CAM (interpretabilidade)|
| 9 | Análise clínica (threshold Stricture + Biliary_Leaks)|

**Para obter os resultados completos**, repete as Secções 1–6 mudando o CONFIG para cada modelo:

```python
# Correr 1: DeiT-III
MODEL_TYPE = 'deit3';       MODEL_PTH = './models/deitiii.pth'

# Correr 2: DenseNet121
MODEL_TYPE = 'densenet';    MODEL_PTH = './models/densenet.pth'

# Correr 3: ResNet50
MODEL_TYPE = 'resnet';      MODEL_PTH = './models/resnet50.pth'

# Correr 4: EfficientNet-B7
MODEL_TYPE = 'efficientnet'; MODEL_PTH = './models/efficientnet_b7.pth'

# Correr 5: MobileNet-V2
MODEL_TYPE = 'mobilenet';   MODEL_PTH = './models/mobilenet_v2.pth'
```

Depois corre a Secção 7 (ensemble) e a Secção 9 (análise clínica).

---

## Resultados Obtidos

### Tabela comparativa por modelo (test set)

| Modelo | CNN puro F1 | XGBoost F1 | Melhor variante |
|--------|------------|------------|----------------|
| DenseNet121 | 0.5353 | **0.6207** | XGBoost |
| ResNet50 | 0.5682 | **0.7035** | XGBoost |
| EfficientNet-B7 | **0.7381** | 0.6949 | CNN |
| MobileNet-V2 | 0.5842 | **0.5894** | XGBoost |
| DeiT-III Small | - | - | - |

### Ensemble misto (4 modelos)

| Métrica | Valor |
|---------|-------|
| **F1 macro** | **0.7797** |
| Accuracy | 0.7978 |
| Baseline paper | 0.738 |

### Análise clínica — Stricture (Secção 9)

Threshold ajustado para 0.20 (em vez do argmax padrão):

| | Recall Stricture | F1 macro |
|-|-----------------|---------|
| Argmax (padrão) | 0.7024 | 0.7797 |
| Threshold = 0.20 | **0.9048** | 0.7934 |

> A Stricture é clinicamente a classe mais perigosa (pode mascarar tumores). Aumentar o recall de 70% para 90% sem custo de F1 macro justifica-se clinicamente.

---

## Interpretabilidade (Grad-CAM)

As figuras Grad-CAM estão em `notebooks/preds/*_gradcam.png`.  
O notebook usa o último bloco convolucional de cada CNN para gerar mapas de activação.

---

## Notas Técnicas

**Split patient-aware vs image-level:**  
O split isola pacientes entre train/val/test. Splits image-level (onde imagens do mesmo paciente aparecem em train e test) inflacionam artificialmente as métricas. O nosso split é mais rigoroso e mais próximo de um cenário clínico real.

**TTA (Test-Time Augmentation):**  
As previsões CNN usam TTA com 4 versões (original + flip horizontal + rot±90°), implementado na função `cnn_predict(use_tta=True)`.

**Desbalanceamento de classes:**  
Tratado com `compute_sample_weight('balanced')` no XGBoost e FocalLoss no treino dos backbones.

---