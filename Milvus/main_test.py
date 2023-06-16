from Tests.ivf_flat import test_accuracy_and_memory, test_recall
from Base.embedding import EmbeddingHandler

if __name__ == "__main__":
    embedding_handler = EmbeddingHandler()
    test_accuracy_and_memory(embedding_handler=embedding_handler)
    # test_recall(embedding_handler=embedding_handler)
