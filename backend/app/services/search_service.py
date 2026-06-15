"""Embeds a query image and looks it up against the per-model vector store."""

from PIL import Image

from ai_services.embeddings import get_model
from ai_services.preprocessing import resize_max_dim

from .vector_stores import get_vector_store


def search_image(image: Image.Image, model_name: str, k: int = 5) -> dict:
    model = get_model(model_name)
    image = resize_max_dim(image)
    vector = model.embed(image)

    vector_store = get_vector_store(model_name)
    matches = vector_store.search(vector, k=k)

    return {
        "model": model_name,
        "matches": [
            {"id": match.id, "score": match.score, "metadata": match.metadata} for match in matches
        ],
        "embedding_map": vector_store.project_2d(),
    }
