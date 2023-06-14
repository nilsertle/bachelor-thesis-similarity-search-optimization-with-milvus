from create import test_ivf_flat, test_recall
from embedding import EmbeddingHandler

if __name__ == "__main__":
    embedding_handler = EmbeddingHandler(entity_count=10, test_count=10)
    # test_ivf_flat(embedding_handler=embedding_handler)
    test_recall(embedding_handler=embedding_handler)
