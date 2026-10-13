"""Download and cache all embedding models (GloVe-50 and SentenceTransformer models)."""
from retrieval.embedders import glove_prepare, ST_MODELS
from sentence_transformers import SentenceTransformer

def main():
    print("=== Downloading & Preparing All Models ===")
    
    print("\n[1/4] Preparing GloVe 50d (69 MB from GitHub)...")
    glove_prepare(50, report=lambda p, msg: print(f"  [{int(p*100)}%] {msg}"))
    print("GloVe 50d ready.")

    for i, spec in enumerate(ST_MODELS, start=2):
        hub_name = spec[2]
        print(f"\n[{i}/4] Downloading {hub_name} ({spec[5]})...")
        model = SentenceTransformer(hub_name)
        dim = model.get_sentence_embedding_dimension()
        print(f"Loaded {hub_name} (dim: {dim}) successfully.")

    print("\n=== All models downloaded and cached successfully! ===")

if __name__ == "__main__":
    main()
