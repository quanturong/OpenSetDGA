from __future__ import annotations
import pathlib
import pandas as pd

DATA_DIR = pathlib.Path(__file__).parent.parent / "data" / "processed"


def _load(name: str) -> pd.DataFrame:
    p = DATA_DIR / "known" / f"{name}.csv"
    if not p.exists():
        p = DATA_DIR / f"{name}.csv"
    return pd.read_csv(p)


def _fam_col(df: pd.DataFrame) -> str:
    return "family" if "family" in df.columns else "label"


def check_no_domain_leakage() -> None:
    train   = _load("train")
    val     = _load("val")
    test    = _load("test_known")
    unk_fam = pd.read_csv(DATA_DIR / "unknown_family" / "test_unknown_family.csv")
    unk_ood = pd.read_csv(DATA_DIR / "unknown_ood" / "test_unknown_ood.csv")

    split_sets = {
        "train":          set(train["domain"]),
        "val":            set(val["domain"]),
        "test_known":     set(test["domain"]),
        "unknown_family": set(unk_fam["domain"]),
        "unknown_ood":    set(unk_ood["domain"]),
    }

    pairs = [
        ("train",          "val"),
        ("train",          "test_known"),
        ("val",            "test_known"),
        ("train",          "unknown_family"),
        ("train",          "unknown_ood"),
        ("val",            "unknown_family"),
        ("val",            "unknown_ood"),
        ("test_known",     "unknown_family"),
        ("test_known",     "unknown_ood"),
        ("unknown_family", "unknown_ood"),
    ]

    any_leak = False
    for a, b in pairs:
        overlap = split_sets[a] & split_sets[b]
        if overlap:
            print(f"[LEAKAGE] {len(overlap)} domains shared between {a} and {b}")
            any_leak = True

    assert not any_leak, "Domain leakage detected — see [LEAKAGE] lines above."
    print("[OK] No domain leakage between any pair of splits.")


def check_known_splits_have_19_classes() -> None:
    train = _load("train")
    val   = _load("val")
    test  = _load("test_known")
    fc = _fam_col(train)

    n_train      = train[fc].nunique()
    val_classes  = set(val[fc].unique())
    test_classes = set(test[fc].unique())
    train_classes = set(train[fc].unique())

    assert n_train == 19, f"[CLASS COUNT] train has {n_train} classes, expected 19"
    assert val_classes  <= train_classes, \
        f"[CLASS LEAK] val has classes not in train: {val_classes - train_classes}"
    assert test_classes <= train_classes, \
        f"[CLASS LEAK] test has classes not in train: {test_classes - train_classes}"
    print("[OK] Known splits: train has 19 classes, val/test_known are valid subsets.")


def check_split_sizes() -> None:
    train   = _load("train")
    val     = _load("val")
    test    = _load("test_known")
    unk_fam = pd.read_csv(DATA_DIR / "unknown_family" / "test_unknown_family.csv")
    unk_ood = pd.read_csv(DATA_DIR / "unknown_ood" / "test_unknown_ood.csv")

    assert len(train)   >= 100_000, f"[SIZE] train too small: {len(train)}"
    assert len(val)     >= 10_000,  f"[SIZE] val too small: {len(val)}"
    assert len(test)    >= 10_000,  f"[SIZE] test_known too small: {len(test)}"
    assert len(unk_fam) >= 10_000,  f"[SIZE] unknown_family too small: {len(unk_fam)}"
    assert len(unk_ood) >= 10_000,  f"[SIZE] unknown_ood too small: {len(unk_ood)}"
    print(f"[OK] Split sizes: train={len(train):,}  val={len(val):,}  "
          f"test={len(test):,}  unk_fam={len(unk_fam):,}  unk_ood={len(unk_ood):,}")


def check_no_conflicting_labels() -> None:
    for name in ("train", "val", "test_known"):
        df = _load(name)
        fc = _fam_col(df)
        conflicts = df.groupby("domain")[fc].nunique()
        conflicts = conflicts[conflicts > 1]
        assert len(conflicts) == 0, (
            f"[CONFLICT] {len(conflicts)} domains with conflicting labels in {name}: "
            f"{conflicts.index.tolist()[:5]}"
        )
    print("[OK] No conflicting labels in any known split.")


def check_near_empty_classes(min_samples: int = 50) -> None:
    train = _load("train")
    fc = _fam_col(train)
    counts = train[fc].value_counts()
    small = counts[counts < min_samples]
    if len(small) > 0:
        print(f"[WARN] {len(small)} classes with < {min_samples} samples in train:")
        print(small.to_string())
    else:
        print(f"[OK] All classes have >= {min_samples} samples.")


def run_all() -> None:
    print("=" * 60)
    print("  Data Assertions")
    print("=" * 60)
    checks = [
        check_no_domain_leakage,
        check_no_conflicting_labels,
        check_known_splits_have_19_classes,
        check_split_sizes,
        check_near_empty_classes,
    ]
    results = []
    for fn in checks:
        try:
            fn()
            results.append(("PASS", fn.__name__))
        except AssertionError as e:
            print(f"[FAIL] {e}")
            results.append(("FAIL", fn.__name__))
    print("\n" + "=" * 60)
    passed = sum(1 for r, _ in results if r == "PASS")
    print(f"  {passed}/{len(results)} assertions passed")
    if passed < len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    run_all()
