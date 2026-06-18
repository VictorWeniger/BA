import numpy as np


def upper_triangle_values(matrix):
    """Oberes Dreieck einer quadratischen Matrix ohne Diagonale.

    RDMs sind symmetrisch — jedes Paar (i,j) taucht nur einmal auf.
    Die Diagonale ist immer 0 (Distanz zu sich selbst) und wird weggelassen.
    """
    rows, cols = np.triu_indices_from(matrix, k=1)
    return matrix[rows, cols]


def rankdata_average_ties(values):
    """Rangwerte berechnen, Gleichstände werden gemittelt.

    Eigene Implementierung ohne scipy — ersetzt scipy.stats.rankdata.
    Ränge beginnen bei 1. Bei Gleichstand: Durchschnittsrang aller Gleichwerte.
    Wird intern von spearman_corr verwendet.
    """
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)

    start = 0
    while start < len(values):
        end = start + 1
        # Alle gleichen Werte finden
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        # Durchschnittsrang für die Gruppe zuweisen
        average_rank = 0.5 * (start + 1 + end)
        ranks[order[start:end]] = average_rank
        start = end

    return ranks


def pearson_corr(x, y):
    """Pearson-Korrelation (linearer Zusammenhang zwischen x und y).

    Basisoperation — wird von spearman_corr auf die Rangwerte angewendet.
    NaN-Werte werden automatisch ausgeschlossen.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # Paare entfernen wo einer der Werte NaN oder Inf ist
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan

    # Zentrierung und normierte Kovarianz
    x = x - x.mean()
    y = y - y.mean()
    denominator = np.sqrt(np.sum(x**2) * np.sum(y**2))
    if denominator == 0:
        return np.nan
    return float(np.sum(x * y) / denominator)


def spearman_corr(x, y):
    """Spearman-Rangkorrelation für RSA.

    Misst monotonen (nicht nur linearen) Zusammenhang.
    Invariant gegenüber monotonen Transformationen der Daten.
    In der RSA: Vergleich der Rangordnung paarweiser Distanzen zwischen DNN und Mensch.
    """
    return pearson_corr(rankdata_average_ties(x), rankdata_average_ties(y))


def permutation_test_spearman(x, y, n_permutations=1000, rng=None):
    """Zweiseitiger Permutationstest für die Spearman-Korrelation.

    Logik: Wenn es keinen echten Zusammenhang gibt, sollte eine zufällige
    Umsortierung von y eine ähnlich hohe Korrelation ergeben.
    p-Wert = Anteil der Permutationen, die mindestens so hoch korrelieren wie
    die beobachtete Korrelation.

    +1 im Zähler und Nenner (Laplace-Korrektur) verhindert p=0, was bei
    endlich vielen Permutationen nicht interpretierbar ist.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    observed = abs(spearman_corr(x, y))
    count = sum(
        abs(spearman_corr(x, rng.permutation(y))) >= observed
        for _ in range(n_permutations)
    )
    return (count + 1) / (n_permutations + 1)


def kernel_cka(K, L):
    """Centered Kernel Alignment für vorberechnete Kernel-Matrizen (Kornblith et al. 2019).

    CKA(K, L) = HSIC(K, L) / sqrt(HSIC(K,K) * HSIC(L,L))

    HSIC misst statistische Abhängigkeit zwischen zwei Kernel-Matrizen.
    Zentrierung mit H = I - 1/n * 11ᵀ entfernt den Mittelwert aus dem Kernel-Raum.
    CKA ∈ [0, 1]: 0 = unabhängig, 1 = identische Struktur.
    Invariant gegenüber orthogonalen Transformationen und isotroper Skalierung.

    Verwendung hier: K_DNN  = Kosinus-Ähnlichkeit der ResNet-RDM,
                     K_fMRI = 1 - fMRI-ROI-Distanzmatrix
    """
    n = K.shape[0]
    # Zentrierungsmatrix H
    H = np.eye(n) - np.ones((n, n)) / n
    K_c = H @ K @ H
    L_c = H @ L @ H
    # HSIC-Schätzer (biased, aber konsistent für große n)
    hsic_kl = np.trace(K_c @ L_c) / (n - 1) ** 2
    hsic_kk = np.trace(K_c @ K_c) / (n - 1) ** 2
    hsic_ll = np.trace(L_c @ L_c) / (n - 1) ** 2
    denom = np.sqrt(hsic_kk * hsic_ll)
    if denom == 0:
        return np.nan
    return float(hsic_kl / denom)


def linear_cka(X, Y):
    """Lineares CKA direkt auf Feature-Matrizen X (n×p) und Y (n×q).

    Äquivalent zu kernel_cka mit linearen Kernen K=XXᵀ, L=YYᵀ —
    aber effizienter wenn p,q << n, weil keine n×n Gram-Matrizen gebildet werden.

    Verwendet für Layer×Layer CKA (wie ähnlich sind sich Schichten untereinander?).
    """
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    # Der 1/(n-1)^2 Faktor kürzt sich im Verhältnis heraus
    hsic_xy = np.linalg.norm(X.T @ Y, "fro") ** 2
    hsic_xx = np.linalg.norm(X.T @ X, "fro") ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, "fro") ** 2
    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom == 0:
        return np.nan
    return float(hsic_xy / denom)
