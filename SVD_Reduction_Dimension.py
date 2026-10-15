import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.datasets import face

# ==========================================
# 1. Chargement et Préparation des Données
# ==========================================
# Chargement d'une image matricielle de test (Convertie en niveaux de gris)
img = face(gray=True)
A = img.astype(float)

# ==========================================
# 2. Décomposition en Valeurs Singulières (SVD)
# Équation : A = U * Sigma * V^T
# ==========================================
U, S, Vt = np.linalg.svd(A, full_matrices=False)

# ==========================================
# 3. Reconstruction Approchée de Rang k
# ==========================================
def reconstruire_image(U, S, Vt, k):
    """
    Reconstruit la matrice A à partir des k premières valeurs singulières.
    """
    return np.dot(U[:, :k], np.dot(np.diag(S[:k]), Vt[:k, :]))

# Valeurs de k pour l'étude de compression
k_valeurs = [5, 20, 50, 100]

# Calcul de l'erreur de norme de Frobenius pour chaque k
erreur_frobenius = []
for k in range(1, 101):
    A_k = reconstruire_image(U, S, Vt, k)
    err = np.linalg.norm(A - A_k, 'fro') / np.linalg.norm(A, 'fro')
    erreur_frobenius.append(err)

# ==========================================
# 4. Visualisation des Résultats (Matplotlib)
# ==========================================
fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))

# Titre principal avec espacement supérieur suffisant
fig.suptitle("Analyse SVD : Compression d'Image et Erreur de Frobenius", 
             fontsize=14, fontweight='bold', y=0.98)

# Image originale
axes[0, 0].imshow(A, cmap='gray')
axes[0, 0].set_title(f"Image Originale\n(Rang max = {min(A.shape)})", fontsize=10, pad=10)
axes[0, 0].axis('off')

# Images reconstruites
coords = [(0, 1), (0, 2), (1, 0), (1, 1)]
for idx, k in enumerate(k_valeurs):
    r, c = coords[idx]
    A_k = reconstruire_image(U, S, Vt, k)
    axes[r, c].imshow(A_k, cmap='gray')
    axes[r, c].set_title(f"Reconstruction k = {k} valeurs", fontsize=10, pad=10)
    axes[r, c].axis('off')

# Graphique de la courbe de l'erreur de Frobenius
axes[1, 2].plot(range(1, 101), erreur_frobenius[:100], color='#2b5c8f', linewidth=2)
axes[1, 2].set_title("Erreur relative de Frobenius vs k", fontsize=10, pad=10)
axes[1, 2].set_xlabel("Rang k", fontsize=9)
axes[1, 2].set_ylabel("Erreur Relative", fontsize=9)
axes[1, 2].grid(True, linestyle=':', alpha=0.6)

# Ajustement des espaces entre les sous-graphiques
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.subplots_adjust(wspace=0.28, hspace=0.35)

# ==========================================
# 5. Sauvegarde et Affichage
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
image_path = os.path.join(script_dir, 'svd_compression_results.png')

plt.savefig(image_path, dpi=300, bbox_inches='tight')
print(f"Image sauvegardée avec succès sous :\n{image_path}")

plt.show()