import numpy as np

from ai_services.vector_store.faiss_store import FaissVectorStore


def test_add_and_search_returns_k_nearest(tmp_path):
    store = FaissVectorStore(dim=8, path=str(tmp_path))
    rng = np.random.default_rng(0)

    for i in range(5):
        store.add(f"item-{i}", rng.standard_normal(8).astype(np.float32), {"label": f"case {i}"})

    results = store.search(rng.standard_normal(8).astype(np.float32), k=3)

    assert len(results) == 3
    assert all(-1.0 - 1e-3 <= match.score <= 1.0 + 1e-3 for match in results)
    assert all(match.id.startswith("item-") for match in results)


def test_search_on_empty_store_returns_no_matches(tmp_path):
    store = FaissVectorStore(dim=8, path=str(tmp_path))

    assert store.search(np.ones(8, dtype=np.float32), k=3) == []


def test_project_2d_returns_a_point_per_item(tmp_path):
    store = FaissVectorStore(dim=8, path=str(tmp_path))
    rng = np.random.default_rng(1)

    for i in range(4):
        store.add(f"item-{i}", rng.standard_normal(8).astype(np.float32), {"label": f"case {i}"})

    points = store.project_2d()

    assert len(points) == 4
    assert all({"id", "x", "y", "label"} <= point.keys() for point in points)


def test_persist_and_reload_round_trip(tmp_path):
    store = FaissVectorStore(dim=8, path=str(tmp_path))
    rng = np.random.default_rng(2)

    for i in range(5):
        store.add(f"item-{i}", rng.standard_normal(8).astype(np.float32), {"label": f"case {i}"})
    store.persist()

    reloaded = FaissVectorStore(dim=8, path=str(tmp_path))

    assert reloaded.index.ntotal == 5
    assert reloaded.ids == store.ids
    assert reloaded.search(rng.standard_normal(8).astype(np.float32), k=2)
