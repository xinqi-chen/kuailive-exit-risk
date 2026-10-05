import csv
import hashlib
from pathlib import Path


def selected_users(path: Path, n_users: int) -> list[str]:
    with path.open(newline="") as f:
        values = {row["user_id"].strip() for row in csv.DictReader(f)}
    if not values or "" in values:
        raise ValueError("invalid user_id_set.csv")
    return sorted(values, key=lambda x: (hashlib.sha256(x.encode()).digest(), x))[:n_users]


def cohort_hash(users: list[str]) -> str:
    return hashlib.sha256(("\n".join(users) + "\n").encode()).hexdigest()
