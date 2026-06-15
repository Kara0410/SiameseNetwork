def test_models_lists_all_backends_and_default(client):
    response = client.get("/api/v1/models")

    assert response.status_code == 200
    data = response.json()
    assert data["default"] == "dummy"

    names = {model["name"] for model in data["models"]}
    assert names == {"dummy", "clip", "vit", "efficientnet", "siamese"}

    dummy = next(model for model in data["models"] if model["name"] == "dummy")
    assert dummy["dimension"] == 64
