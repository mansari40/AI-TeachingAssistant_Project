from src.utils.ids import make_chunk_id, make_point_id


def test_chunk_id_stability() -> None:
    a = make_chunk_id("pdf", "doc1", 0)
    b = make_chunk_id("pdf", "doc1", 0)
    assert a == b


def test_point_id_deterministic() -> None:
    cid = "pdf::doc1::0"
    a = make_point_id(cid)
    b = make_point_id(cid)
    assert a == b
