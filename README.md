# Mini-LLM-Efficient-GPT-Style-Language-Model

Un modèle de langage GPT-like efficace implémenté from scratch, entraîné sur 200,000 histoires avec une architecture Transformer optimisée. Parfait pour la génération de texte créatif et l'apprentissage des LLMs.

# Fonctionnalités Principales

#  Architecture Avancée
- **Transformer Decoder-only** (style GPT) avec 8 couches
- **25.4 millions de paramètres** optimisés
- **Multi-Head Attention** (8 têtes) avec causal masking
- **Weight Tying** entre les embeddings et la couche de sortie
- **Character-level tokenization** avec vocabulaire de 96 tokens

#  Performance & Optimisation
- **Mixed Precision Training** (FP16) pour un entraînement accéléré
- **Gradient Clipping** et **Learning Rate Scheduling** (cosine + warmup)
- **Validation Loss**: 0.40 sur le dataset de validation
- **Génération autoregressive** avec top-k sampling et contrôle de température

# Génération Intelligente
#Contrôle fin de la génération
story = generator.generate_story(
    prompt="Once upon a time",
    max_length=200,      # 50-500 tokens
    temperature=0.8,     # 0.5-1.5 (créativité)
    top_k=45            # 10-100 (diversité)
)

# Algorithmes de Génération
Autoregressive avec causal masking
Top-k Sampling (k=10-100) pour éviter la répétition
Temperature Scaling (0.5-1.5) pour contrôler la créativité
Longueur configurable de 50 à 500 tokens



# Installer les dépendances
pip install -r requirements.txt

# Demo
<img width="900" height="742" alt="Capture d’écran 2025-10-28 154228" src="https://github.com/user-attachments/assets/36ab953a-0c92-42e8-be03-c6280452595e" />

<img width="1887" height="872" alt="Capture d’écran 2025-10-28 153942" src="https://github.com/user-attachments/assets/69963840-b2ec-4024-bb80-5456a562c8b9" />
<img width="891" height="756" alt="Capture d’écran 2025-10-28 154044" src="https://github.com/user-attachments/assets/499e1f76-1868-401c-b2a3-0b3d68bb9c19" />
<img width="917" height="671" alt="Capture d’écran 2025-10-28 154334" src="https://github.com/user-attachments/assets/fa842b9b-b655-4755-afe0-a549637b24be" />

