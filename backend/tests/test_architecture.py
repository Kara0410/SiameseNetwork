def test_architecture_returns_pipeline_latency_and_deployment(client):
    response = client.get("/api/v1/architecture")

    assert response.status_code == 200
    data = response.json()

    assert len(data["pipeline"]) == 6
    assert {stage["id"] for stage in data["pipeline"]} == {
        "ingest",
        "encoder",
        "metric",
        "retrieval",
        "reasoning",
        "serve",
    }

    assert {entry["stage"] for entry in data["latency"]} == {
        "capture",
        "encode",
        "search",
        "reason",
        "api",
    }
    assert all(entry["ms"] > 0 for entry in data["latency"])

    assert data["deployment"]
    assert all({"category", "items"} <= group.keys() for group in data["deployment"])
