import os
import glob
import logging
import warnings
import time
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.base import clone

import smote_variants as sv
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.metrics import geometric_mean_score
from tqdm.notebook import tqdm

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.WARNING)
logging.getLogger("smote_variants").setLevel(logging.ERROR)
logging.getLogger("NearestNeighborsWithMetricTensor").setLevel(logging.ERROR)
for name in logging.root.manager.loggerDict:
    if "smote" in name.lower() or "neighbor" in name.lower():
        logging.getLogger(name).setLevel(logging.ERROR)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DATASET_ROOT = "./datasets"
DETAILED_OUTPUT_EXCEL = "benchmark_results_pool_ablation.xlsx"
SUMMARY_OUTPUT_EXCEL = "benchmark_summary_pool_ablation.xlsx"

AUGMENTATION_RATES = [0.25, 0.50, 0.75, 1.00]

# =============================================================================
# METHOD SETS
# =============================================================================
# Klasik karşılaştırma seti (user-defined baseline set)
# Not:
# - Original ve ROS aşağıda apply_classic_oversamplers içinde ayrıca üretilir.
# - SMOTEENN ve SMOTETomek de ayrıca imblearn ile üretilir.
CLASSIC_OVERSAMPLING_METHODS = {
    "SMOTE": sv.SMOTE,
    "ADASYN": sv.ADASYN,
    "Borderline_SMOTE1": sv.Borderline_SMOTE1,
    "Borderline_SMOTE2": sv.Borderline_SMOTE2,
    "MWMOTE": sv.MWMOTE,
    "Safe_Level_SMOTE": sv.Safe_Level_SMOTE,
    "G_SMOTE": sv.G_SMOTE,
    "kmeans_SMOTE": sv.kmeans_SMOTE,
}

CLASSIC_METHODS_ALL = [
    "Original",
    "ROS",
    "SMOTE",
    "ADASYN",
    "Borderline_SMOTE1",
    "Borderline_SMOTE2",
    "MWMOTE",
    "Safe_Level_SMOTE",
    "G_SMOTE",
    "kmeans_SMOTE",
    "SMOTEENN",
    "SMOTETomek",
]

# Havuz üretiminde kullanılabilecek jeneratörler.
# Ayrı tutmamızın sebebi:
# - Baseline listesi ile pool listesi kavramsal olarak farklıdır.
# - İleride bir yöntemi sadece pool tarafına veya sadece baseline tarafına eklemek kolaylaşır.
POOL_GENERATOR_METHODS = {
    **CLASSIC_OVERSAMPLING_METHODS,
}

# Havuz senaryoları: önerilen yöntemler bu jeneratörlerle synthetic pool oluşturur.
# Yeni eklenen senaryo:
#   Pool_SMOTE_ADASYN_MWMOTE_G_SMOTE = SMOTE + ADASYN + MWMOTE + G_SMOTE
POOL_SCENARIOS = {
    "Pool_All": [
        "SMOTE", "Borderline_SMOTE1", "ADASYN", "MWMOTE",
        "kmeans_SMOTE", "Safe_Level_SMOTE"
    ],
    "Pool_Drop_kmeans": [
        "SMOTE", "Borderline_SMOTE1", "ADASYN", "MWMOTE", "Safe_Level_SMOTE"
    ],
    "Pool_Drop_kmeans_Borderline": [
        "SMOTE", "ADASYN", "MWMOTE", "Safe_Level_SMOTE"
    ],
    "Pool_Drop_kmeans_Borderline_Safe": [
        "SMOTE", "ADASYN", "MWMOTE"
    ],
    "Pool_Core": [
        "SMOTE", "ADASYN"
    ],
    "Pool_Strong_Classics": [
        "SMOTE", "ADASYN", "SMOTEENN", "SMOTETomek"
    ],
    "Pool_SMOTE_ADASYN_MWMOTE_G_SMOTE": [
        "SMOTE", "ADASYN", "MWMOTE", "G_SMOTE"
    ],
}


# =============================================================================
# DATA IO
# =============================================================================

def parse_keel_dat(filepath):
    lines = []
    data_section = False
    attribute_names = []
    output_attr = None

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue

            low = line.lower()
            if low.startswith("@outputs"):
                output_attr = line.split()[-1].strip()
            elif low.startswith("@attribute"):
                parts = line.split()
                attribute_names.append(parts[1])
            elif low.startswith("@data"):
                data_section = True
            elif data_section:
                lines.append(line)

    rows = [l.split(",") for l in lines]
    df = pd.DataFrame(rows, columns=attribute_names)
    df.columns = df.columns.str.strip()

    label_col = output_attr.strip() if output_attr else df.columns[-1]

    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()

    X_df = df.drop(columns=[label_col]).copy()
    y_raw = df[label_col].copy()

    for col in X_df.columns:
        converted = pd.to_numeric(X_df[col], errors="coerce")
        if converted.notna().all():
            X_df[col] = converted.astype(float)
        else:
            X_df[col] = LabelEncoder().fit_transform(X_df[col].astype(str))

    for col in X_df.columns:
        if X_df[col].isna().any():
            X_df[col] = X_df[col].fillna(X_df[col].median())

    X = X_df.values.astype(np.float64)
    y = LabelEncoder().fit_transform(y_raw.values)

    # Azınlık sınıfını 1 olarak sabitle
    if np.sum(y == 1) > np.sum(y == 0):
        y = 1 - y

    if np.isnan(X).any():
        raise ValueError(f"{filepath} dosyasında parse sonrası NaN kaldı.")

    return X, y


def load_fold(dataset_dir, dataset_name, fold_idx):
    train_file = os.path.join(dataset_dir, f"{dataset_name}-5-{fold_idx}tra.dat")
    test_file = os.path.join(dataset_dir, f"{dataset_name}-5-{fold_idx}tst.dat")
    X_train, y_train = parse_keel_dat(train_file)
    X_test, y_test = parse_keel_dat(test_file)
    return X_train, y_train, X_test, y_test


# =============================================================================
# UTILITIES
# =============================================================================

def get_minority_count(y_train):
    return int(np.sum(y_train == 1))


def get_safe_k_neighbors(y_train, default_k=5, max_safe_k=5):
    minority_count = get_minority_count(y_train)
    if minority_count < 2:
        return None
    k = min(default_k, max_safe_k, minority_count - 1)
    return int(k) if k >= 1 else None


def compute_target_minority(y_train, rate):
    n_majority = int(np.sum(y_train == 0))
    n_minority = int(np.sum(y_train == 1))
    gap = n_majority - n_minority
    needed = max(0, round(rate * gap))
    target = n_minority + needed
    return target, needed


def get_classifier_templates():
    return {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1),
        "SVM_RBF": SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, n_jobs=-1),
        "GaussianNB": GaussianNB(),
    }


def get_classifiers():
    return {name: clone(model) for name, model in get_classifier_templates().items()}


def evaluate_classifier(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    try:
        y_prob = clf.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
    except Exception:
        roc_auc = np.nan

    return {
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "GMean": geometric_mean_score(y_test, y_pred, average="binary"),
        "BalancedAccuracy": balanced_accuracy_score(y_test, y_pred),
        "ROC_AUC": roc_auc,
    }


# =============================================================================
# CLASSIC OVERSAMPLERS (FULL TRAIN SET OUTPUT)
# =============================================================================

def _fit_sv_oversampler(method_class, X_train, y_train, proportion=None):
    safe_k = get_safe_k_neighbors(y_train)
    if safe_k is None:
        return X_train.copy(), y_train.copy(), False, f"minority örnek sayısı çok az ({get_minority_count(y_train)})"

    candidate_param_sets = [
        {"proportion": proportion, "random_state": RANDOM_SEED, "n_neighbors": safe_k, "k_neighbors": safe_k},
        {"proportion": proportion, "random_state": RANDOM_SEED, "k_neighbors": safe_k},
        {"proportion": proportion, "random_state": RANDOM_SEED, "n_neighbors": safe_k},
        {"proportion": proportion, "random_state": RANDOM_SEED},
        {"random_state": RANDOM_SEED, "n_neighbors": safe_k, "k_neighbors": safe_k},
        {"random_state": RANDOM_SEED, "k_neighbors": safe_k},
        {"random_state": RANDOM_SEED, "n_neighbors": safe_k},
        {"random_state": RANDOM_SEED},
    ]

    tried = set()
    last_error = None
    for init_params in candidate_param_sets:
        init_params = {k: v for k, v in init_params.items() if v is not None}
        key = tuple(sorted(init_params.items()))
        if key in tried:
            continue
        tried.add(key)
        try:
            oversampler = method_class(**init_params)
            X_res, y_res = oversampler.sample(X_train, y_train)
            return X_res, y_res, True, None
        except Exception as e:
            last_error = e

    return X_train.copy(), y_train.copy(), False, str(last_error)


def apply_classic_oversamplers(X_train, y_train, rate):
    variants = {"Original": (X_train.copy(), y_train.copy())}

    n_majority = int(np.sum(y_train == 0))
    target_min, _ = compute_target_minority(y_train, rate)
    imblearn_strategy = {1: target_min}
    sv_ratio = target_min / n_majority if n_majority > 0 else 1.0
    safe_k = get_safe_k_neighbors(y_train)

    # ROS
    try:
        ros = RandomOverSampler(sampling_strategy=imblearn_strategy, random_state=RANDOM_SEED)
        variants["ROS"] = ros.fit_resample(X_train, y_train)
    except Exception:
        variants["ROS"] = (X_train.copy(), y_train.copy())

    # smote-variants tabanlı klasik yöntemler
    for method_name, method_class in CLASSIC_OVERSAMPLING_METHODS.items():
        X_res, y_res, success, _ = _fit_sv_oversampler(method_class, X_train, y_train, proportion=sv_ratio)
        variants[method_name] = (X_res, y_res) if success else (X_train.copy(), y_train.copy())

    # Hybrid-cleaning yöntemler
    try:
        if safe_k is None:
            raise ValueError("minority örnek sayısı yetersiz")
        smoteenn = SMOTEENN(
            smote=SMOTE(sampling_strategy=imblearn_strategy, k_neighbors=safe_k, random_state=RANDOM_SEED),
            random_state=RANDOM_SEED,
        )
        variants["SMOTEENN"] = smoteenn.fit_resample(X_train, y_train)
    except Exception:
        variants["SMOTEENN"] = (X_train.copy(), y_train.copy())

    try:
        if safe_k is None:
            raise ValueError("minority örnek sayısı yetersiz")
        smotetomek = SMOTETomek(
            smote=SMOTE(sampling_strategy=imblearn_strategy, k_neighbors=safe_k, random_state=RANDOM_SEED),
            random_state=RANDOM_SEED,
        )
        variants["SMOTETomek"] = smotetomek.fit_resample(X_train, y_train)
    except Exception:
        variants["SMOTETomek"] = (X_train.copy(), y_train.copy())

    return variants


# =============================================================================
# SYNTHETIC POOL BUILDERS
# =============================================================================

def _extract_synthetic_minority(X_train, y_train, X_res, y_res):
    original_count = len(X_train)
    new_X = X_res[original_count:]
    new_y = y_res[original_count:]
    mask = new_y == 1
    return new_X[mask], new_y[mask]


def _build_pool_from_sv_generators(X_train, y_train, generator_names, rate):
    target_min, _ = compute_target_minority(y_train, rate)
    n_majority = int(np.sum(y_train == 0))
    sv_ratio = target_min / n_majority if n_majority > 0 else 1.0

    pool_X_list = []
    for method_name in generator_names:
        if method_name not in POOL_GENERATOR_METHODS:
            continue

        method_class = POOL_GENERATOR_METHODS[method_name]
        X_res, y_res, success, _ = _fit_sv_oversampler(method_class, X_train, y_train, proportion=sv_ratio)
        if not success:
            continue

        syn_X, syn_y = _extract_synthetic_minority(X_train, y_train, X_res, y_res)
        if len(syn_X) > 0:
            pool_X_list.append(syn_X)

    if not pool_X_list:
        return np.empty((0, X_train.shape[1])), np.empty((0,), dtype=int)

    pool_X = np.vstack(pool_X_list)
    pool_y = np.ones(len(pool_X), dtype=int)
    return pool_X, pool_y


def _build_pool_from_classic_variants(classic_variants, X_train, y_train, generator_names):
    pool_X_list = []
    for method_name in generator_names:
        if method_name not in classic_variants:
            continue
        X_res, y_res = classic_variants[method_name]
        syn_X, syn_y = _extract_synthetic_minority(X_train, y_train, X_res, y_res)
        if len(syn_X) > 0:
            pool_X_list.append(syn_X)

    if not pool_X_list:
        return np.empty((0, X_train.shape[1])), np.empty((0,), dtype=int)

    pool_X = np.vstack(pool_X_list)
    pool_y = np.ones(len(pool_X), dtype=int)
    return pool_X, pool_y


def build_synthetic_pool(X_train, y_train, rate, generator_names, classic_variants=None):
    if classic_variants is not None:
        # Önce halihazırda üretilmiş klasik varyantlardan yararlan
        effective_generators = [g for g in generator_names if g in classic_variants]
        if effective_generators:
            return _build_pool_from_classic_variants(
                classic_variants, X_train, y_train, effective_generators
            )

    # Kalan jeneratörleri doğrudan smote-variants ile üret
    return _build_pool_from_sv_generators(X_train, y_train, generator_names, rate)


# =============================================================================
# POOL / NEIGHBOR PRECOMPUTATION HELPERS
# =============================================================================

def build_neighbor_context(X_train, y_train):
    """Pre-fit neighbor models once per fold for density/boundary scoring."""
    context = {"eps": 1e-9, "y_train": y_train}

    X_minority = X_train[y_train == 1]
    if len(X_minority) > 1:
        k_min = min(5, len(X_minority) - 1)
        nn_min = NearestNeighbors(n_neighbors=k_min, n_jobs=-1)
        nn_min.fit(X_minority)
        context["nn_min"] = nn_min
    else:
        context["nn_min"] = None

    if len(X_train) > 1:
        k_all = min(5, len(X_train) - 1)
        nn_all = NearestNeighbors(n_neighbors=k_all, n_jobs=-1)
        nn_all.fit(X_train)
        context["nn_all"] = nn_all

        k_boundary = min(10, len(X_train) - 1)
        nn_boundary = NearestNeighbors(n_neighbors=k_boundary, n_jobs=-1)
        nn_boundary.fit(X_train)
        context["nn_boundary"] = nn_boundary
    else:
        context["nn_all"] = None
        context["nn_boundary"] = None

    return context


def precompute_pools_for_scenarios(X_train, y_train, rate, classic_variants):
    """Build each scenario pool once per rate and reuse across proposed selectors."""
    scenario_pools = {}
    for scenario_name, generator_names in POOL_SCENARIOS.items():
        try:
            pool_X, pool_y = build_synthetic_pool(
                X_train, y_train, rate, generator_names, classic_variants=classic_variants
            )
            scenario_pools[scenario_name] = (pool_X, pool_y)
        except Exception:
            scenario_pools[scenario_name] = (
                np.empty((0, X_train.shape[1])),
                np.empty((0,), dtype=int),
            )
    return scenario_pools

# =============================================================================
# PROPOSED METHODS (POOL-SCENARIO AWARE)
# =============================================================================

def _pool_select(X_train, y_train, pool_X, pool_y, scores, needed):
    if len(pool_X) == 0 or needed <= 0:
        return X_train.copy(), y_train.copy()

    scores = np.asarray(scores, dtype=float)
    if len(scores) != len(pool_X):
        raise ValueError("scores uzunluğu ile pool boyutu eşleşmiyor")

    scores = scores - np.nanmin(scores) + 1e-9
    probs = scores / scores.sum()
    replace = len(pool_X) < needed
    rng = np.random.RandomState(RANDOM_SEED)
    idx = rng.choice(len(pool_X), size=needed, replace=replace, p=probs)

    X_aug = np.vstack([X_train, pool_X[idx]])
    y_aug = np.concatenate([y_train, pool_y[idx]])
    return X_aug, y_aug


def proposed_random_from_pool(X_train, y_train, needed, pool_X, pool_y):
    if needed <= 0 or len(pool_X) == 0:
        return X_train.copy(), y_train.copy()

    rng = np.random.RandomState(RANDOM_SEED)
    replace = len(pool_X) < needed
    idx = rng.choice(len(pool_X), size=needed, replace=replace)

    X_aug = np.vstack([X_train, pool_X[idx]])
    y_aug = np.concatenate([y_train, pool_y[idx]])
    return X_aug, y_aug


def proposed_density_from_pool(X_train, y_train, needed, pool_X, pool_y, neighbor_context):
    if needed <= 0 or len(pool_X) == 0:
        return X_train.copy(), y_train.copy()

    nn_min = neighbor_context.get("nn_min")
    nn_all = neighbor_context.get("nn_all")
    eps = neighbor_context.get("eps", 1e-9)

    if nn_min is None or nn_all is None:
        return X_train.copy(), y_train.copy()

    dist_min, _ = nn_min.kneighbors(pool_X)
    proximity = 1.0 / (dist_min.mean(axis=1) + eps)

    _, all_idx = nn_all.kneighbors(pool_X)
    y_neighbors = y_train[all_idx]
    majority_ratio = (y_neighbors == 0).mean(axis=1)

    scores = proximity / (majority_ratio + eps)
    return _pool_select(X_train, y_train, pool_X, pool_y, scores, needed)


def proposed_diversity_from_pool(X_train, y_train, needed, pool_X, pool_y):
    if needed <= 0 or len(pool_X) == 0:
        return X_train.copy(), y_train.copy()

    n_clusters = min(max(2, needed // 5), max(1, len(pool_X) // 2), 20)
    n_clusters = max(1, n_clusters)
    rng = np.random.RandomState(RANDOM_SEED)

    if n_clusters == 1 or len(pool_X) <= n_clusters:
        replace = len(pool_X) < needed
        idx = rng.choice(len(pool_X), size=needed, replace=replace)
        return np.vstack([X_train, pool_X[idx]]), np.concatenate([y_train, pool_y[idx]])

    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
    cluster_labels = km.fit_predict(pool_X)

    per_cluster = needed // n_clusters
    remainder = needed % n_clusters

    cluster_sizes = [(np.sum(cluster_labels == c), c) for c in range(n_clusters)]
    cluster_sizes.sort(reverse=True)

    selected_idx = []
    for rank, (size, c) in enumerate(cluster_sizes):
        c_idx = np.where(cluster_labels == c)[0]
        take = per_cluster + (1 if rank < remainder else 0)
        if take <= 0:
            continue
        take = min(take, len(c_idx))
        chosen = rng.choice(c_idx, size=take, replace=(take > len(c_idx)))
        selected_idx.extend(chosen.tolist())

    while len(selected_idx) < needed:
        selected_idx.append(rng.randint(0, len(pool_X)))

    selected_idx = np.array(selected_idx[:needed])
    X_aug = np.vstack([X_train, pool_X[selected_idx]])
    y_aug = np.concatenate([y_train, pool_y[selected_idx]])
    return X_aug, y_aug


def proposed_boundary_from_pool(X_train, y_train, needed, pool_X, pool_y, neighbor_context):
    if needed <= 0 or len(pool_X) == 0:
        return X_train.copy(), y_train.copy()

    nn_boundary = neighbor_context.get("nn_boundary")
    if nn_boundary is None:
        return X_train.copy(), y_train.copy()

    _, neighbors_idx = nn_boundary.kneighbors(pool_X)
    y_neighbors = y_train[neighbors_idx]

    p_min = (y_neighbors == 1).mean(axis=1)
    p_maj = (y_neighbors == 0).mean(axis=1)
    boundary_score = 1.0 - np.abs(p_min - p_maj)
    boundary_score = np.clip(boundary_score, 1e-9, None)

    return _pool_select(X_train, y_train, pool_X, pool_y, boundary_score, needed)


# =============================================================================
# VARIANT GENERATION
# =============================================================================

def _build_variants_with_pool_scenarios(X_train, y_train, rate, neighbor_context):
    variants = {}
    _, needed = compute_target_minority(y_train, rate)

    classic_variants = apply_classic_oversamplers(X_train, y_train, rate)
    variants.update(classic_variants)

    scenario_pools = precompute_pools_for_scenarios(X_train, y_train, rate, classic_variants)

    for scenario_name, (pool_X, pool_y) in scenario_pools.items():
        try:
            X_rand, y_rand = proposed_random_from_pool(X_train, y_train, needed, pool_X, pool_y)
            variants[f"Proposed_Random__{scenario_name}"] = (X_rand, y_rand)
        except Exception:
            variants[f"Proposed_Random__{scenario_name}"] = (X_train.copy(), y_train.copy())

        try:
            X_den, y_den = proposed_density_from_pool(X_train, y_train, needed, pool_X, pool_y, neighbor_context)
            variants[f"Proposed_Density__{scenario_name}"] = (X_den, y_den)
        except Exception:
            variants[f"Proposed_Density__{scenario_name}"] = (X_train.copy(), y_train.copy())

        try:
            X_div, y_div = proposed_diversity_from_pool(X_train, y_train, needed, pool_X, pool_y)
            variants[f"Proposed_Diversity__{scenario_name}"] = (X_div, y_div)
        except Exception:
            variants[f"Proposed_Diversity__{scenario_name}"] = (X_train.copy(), y_train.copy())

        try:
            X_bnd, y_bnd = proposed_boundary_from_pool(X_train, y_train, needed, pool_X, pool_y, neighbor_context)
            variants[f"Proposed_Boundary__{scenario_name}"] = (X_bnd, y_bnd)
        except Exception:
            variants[f"Proposed_Boundary__{scenario_name}"] = (X_train.copy(), y_train.copy())

    return variants


def _format_seconds(seconds):
    if seconds is None or np.isnan(seconds) or np.isinf(seconds):
        return "?"
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


# =============================================================================
# PIPELINE
# =============================================================================

def run_pipeline(dataset_root=DATASET_ROOT, show_progress=True):
    fold_results = []
    classifier_templates = get_classifier_templates()

    dataset_dirs = sorted([
        d for d in glob.glob(os.path.join(dataset_root, "*")) if os.path.isdir(d)
    ])

    if len(dataset_dirs) == 0:
        raise FileNotFoundError(f"'{dataset_root}' dizininde dataset klasörü bulunamadı.")

    total_steps = len(dataset_dirs) * 5 * len(AUGMENTATION_RATES)
    current_step = 0
    completed_steps = 0
    elapsed_step_time = 0.0

    pbar = tqdm(total=total_steps, desc="Genel İlerleme", unit="adım", disable=not show_progress)

    print(f"Toplam {len(dataset_dirs)} dataset bulundu.")
    print(f"Toplam adım sayısı: {total_steps}\n")
    print(f"Klasik karşılaştırma seti ({len(CLASSIC_METHODS_ALL)} yöntem): {', '.join(CLASSIC_METHODS_ALL)}")
    print(f"Pool senaryosu sayısı: {len(POOL_SCENARIOS)}")
    print(f"Toplam değerlendirilen yöntem sayısı: {len(CLASSIC_METHODS_ALL) + 4 * len(POOL_SCENARIOS)}\n")

    for dataset_dir in dataset_dirs:
        dataset_name = os.path.basename(dataset_dir)
        print("=" * 60)
        print(f"Dataset: {dataset_name}")
        print("=" * 60)

        for fold_idx in range(1, 6):
            print(f"\n  Fold {fold_idx}/5")
            try:
                X_train, y_train, X_test, y_test = load_fold(dataset_dir, dataset_name, fold_idx)
            except Exception as e:
                print(f"  [HATA] Fold {fold_idx} yüklenemedi: {e}")
                skipped = len(AUGMENTATION_RATES)
                current_step += skipped
                if show_progress:
                    pbar.update(skipped)
                    pbar.set_postfix({
                        "dataset": dataset_name,
                        "fold": fold_idx,
                        "rate": "skip",
                        "kalan": total_steps - current_step,
                        "eta": _format_seconds((elapsed_step_time / completed_steps) * (total_steps - current_step))
                        if completed_steps > 0 else "?",
                    })
                continue

            print(f"    Train: {X_train.shape}, Minority: {np.sum(y_train == 1)}, Majority: {np.sum(y_train == 0)}")
            neighbor_context = build_neighbor_context(X_train, y_train)

            for rate in AUGMENTATION_RATES:
                step_start = time.perf_counter()
                current_step += 1
                remaining_steps = total_steps - current_step
                progress_pct = 100 * current_step / total_steps
                _, needed = compute_target_minority(y_train, rate)

                variants = _build_variants_with_pool_scenarios(X_train, y_train, rate, neighbor_context)
                step_model_count = len(variants) * len(classifier_templates)

                avg_step_seconds = (elapsed_step_time / completed_steps) if completed_steps > 0 else None
                eta_text = _format_seconds(avg_step_seconds * remaining_steps) if avg_step_seconds is not None else "?"

                if show_progress:
                    pbar.set_postfix({
                        "dataset": dataset_name,
                        "fold": fold_idx,
                        "rate": f"{int(rate * 100)}%",
                        "kalan": remaining_steps,
                        "eta": eta_text,
                    })

                print(
                    f"    Adım {current_step}/{total_steps} "
                    f"(%{progress_pct:.1f}) | Kalan: {remaining_steps} | "
                    f"Tahmini Süre: {eta_text} | "
                    f"Rate: {int(rate * 100):3d}% (needed={needed}) | Model: {step_model_count}",
                    end=" "
                )

                for method_name, (X_aug, y_aug) in variants.items():
                    for clf_name, template in classifier_templates.items():
                        clf = clone(template)
                        metrics = evaluate_classifier(clf, X_aug, y_aug, X_test, y_test)
                        fold_results.append({
                            "dataset": dataset_name,
                            "fold": fold_idx,
                            "augmentation_rate": rate,
                            "method": method_name,
                            "classifier": clf_name,
                            "F1": metrics["F1"],
                            "GMean": metrics["GMean"],
                            "BalancedAccuracy": metrics["BalancedAccuracy"],
                            "ROC_AUC": metrics["ROC_AUC"],
                        })

                step_elapsed = time.perf_counter() - step_start
                completed_steps += 1
                elapsed_step_time += step_elapsed

                if show_progress:
                    pbar.update(1)
                    new_avg_step_seconds = elapsed_step_time / completed_steps
                    new_eta_text = _format_seconds(new_avg_step_seconds * remaining_steps)
                    pbar.set_postfix({
                        "dataset": dataset_name,
                        "fold": fold_idx,
                        "rate": f"{int(rate * 100)}%",
                        "kalan": remaining_steps,
                        "eta": new_eta_text,
                    })

                print(f"✓ | Adım Süresi: {_format_seconds(step_elapsed)}")

    if show_progress:
        pbar.close()

    return pd.DataFrame(fold_results)


# =============================================================================
# SUMMARIES
# =============================================================================

def compute_dataset_means(fold_df):
    group_cols = ["dataset", "augmentation_rate", "method", "classifier"]
    metric_cols = ["F1", "GMean", "BalancedAccuracy", "ROC_AUC"]
    return fold_df.groupby(group_cols)[metric_cols].mean().reset_index()


def compute_classifier_means(dataset_df):
    group_cols = ["augmentation_rate", "method", "classifier"]
    metric_cols = ["F1", "GMean", "BalancedAccuracy", "ROC_AUC"]
    return dataset_df.groupby(group_cols)[metric_cols].mean().reset_index()


def _winner_total(series):
    if series.isna().all():
        return 0
    best = series.max()
    return int((series == best).sum())


def build_summary_table(fold_df):
    dataset_df = compute_dataset_means(fold_df)
    metrics = ["F1", "GMean", "BalancedAccuracy", "ROC_AUC"]
    summary_rows = []

    for rate in sorted(dataset_df["augmentation_rate"].unique()):
        rate_df = dataset_df[dataset_df["augmentation_rate"] == rate].copy()

        for clf in sorted(rate_df["classifier"].unique()):
            clf_df = rate_df[rate_df["classifier"] == clf].copy()

            for metric in metrics:
                wide = clf_df.pivot_table(index="dataset", columns="method", values=metric, aggfunc="mean")
                if wide.empty:
                    continue

                methods = list(wide.columns)
                avg = wide.mean(axis=0, skipna=True)
                winner_counts = {m: int((wide[m] == wide.max(axis=1)).sum()) for m in methods}
                winner_total = pd.Series(winner_counts)
                ranks = wide.rank(axis=1, method="average", ascending=False)
                mean_rank = ranks.mean(axis=0, skipna=True)
                final_rank = mean_rank.rank(method="dense", ascending=True)

                for method in methods:
                    summary_rows.append({
                        "augmentation_rate": rate,
                        "classifier": clf,
                        "metric": metric,
                        "method": method,
                        "Average": round(float(avg[method]), 6) if pd.notna(avg[method]) else np.nan,
                        "Winner_Total": int(winner_total[method]),
                        "Mean_Rank": round(float(mean_rank[method]), 6) if pd.notna(mean_rank[method]) else np.nan,
                        "Final_Rank": int(final_rank[method]) if pd.notna(final_rank[method]) else np.nan,
                        "Total_Datasets": int(wide.shape[0]),
                    })

    return pd.DataFrame(summary_rows)


def build_overall_method_summary(summary_df):
    if summary_df.empty:
        return pd.DataFrame()

    out = summary_df.groupby("method").agg(
        Average_of_Averages=("Average", "mean"),
        Total_Winner_Count=("Winner_Total", "sum"),
        Mean_of_Mean_Rank=("Mean_Rank", "mean"),
        Mean_of_Final_Rank=("Final_Rank", "mean"),
    ).reset_index()

    out = out.sort_values(["Mean_of_Mean_Rank", "Total_Winner_Count"], ascending=[True, False])
    return out


# =============================================================================
# EXCEL OUTPUT
# =============================================================================

def save_outputs(
    fold_df,
    dataset_df,
    classifier_df,
    summary_df,
    detailed_output=DETAILED_OUTPUT_EXCEL,
    summary_output=SUMMARY_OUTPUT_EXCEL
):
    with pd.ExcelWriter(detailed_output, engine="openpyxl") as writer:
        fold_df.to_excel(writer, sheet_name="Fold Results", index=False)
        dataset_df.to_excel(writer, sheet_name="Dataset Mean Results", index=False)
        classifier_df.to_excel(writer, sheet_name="Classifier Mean Results", index=False)

    overall_summary = build_overall_method_summary(summary_df)
    with pd.ExcelWriter(summary_output, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary Table", index=False)
        overall_summary.to_excel(writer, sheet_name="Overall Method Summary", index=False)

    print(f"\n✅ Detaylı sonuç dosyası kaydedildi: {detailed_output}")
    print(f"✅ Özet tablo dosyası kaydedildi: {summary_output}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("IMBALANCED CLASSIFICATION BENCHMARKING PIPELINE — POOL ABLATION")
    print("Proposed Methods: Random | Density | Diversity | Boundary")
    print("Outputs: Detailed Results + Summary Table (no statistical tests)")
    print("=" * 70)

    fold_df = run_pipeline(dataset_root=DATASET_ROOT)
    if fold_df.empty:
        print("[HATA] Hiç sonuç üretilemedi. Dataset yolunu kontrol edin.")
        return

    print("\nOrtalamalar ve özet tablolar hesaplanıyor...")
    dataset_df = compute_dataset_means(fold_df)
    classifier_df = compute_classifier_means(dataset_df)
    summary_df = build_summary_table(fold_df)

    save_outputs(
        fold_df=fold_df,
        dataset_df=dataset_df,
        classifier_df=classifier_df,
        summary_df=summary_df,
        detailed_output=DETAILED_OUTPUT_EXCEL,
        summary_output=SUMMARY_OUTPUT_EXCEL,
    )


if __name__ == "__main__":
    main()
