from create import test_ivf_flat, test_recall
from embedding import EmbeddingHandler
from Helpers.vecs_helper import ivecs_read, fvecs_read
from base import MilvusHandler    
from Helpers.recall_helper import calc_recall_rate
import numpy as np

def run_tests():
    print("run_tests")
    client = MilvusHandler()
    queries_per_second, time_per_query, recall = client.test_search()
    print(f'Queries per second: {queries_per_second}')
    print(f'Time per query: {time_per_query}')
    print(f'Recall: {recall}')

    

if __name__ == "__main__":
    # embedding_handler = EmbeddingHandler(entity_count=1000, test_count=100)
    # test_ivf_flat(embedding_handler=embedding_handler)
    # test_recall(embedding_handler=embedding_handler)
    # Load base vectors
    # base_file = 'siftsmall/siftsmall_base.fvecs'
    # groundtruth_file = 'siftsmall/siftsmall_groundtruth.ivecs'
    # query_file = 'siftsmall/siftsmall_query.fvecs'

    base_file = 'sift1M/sift_base.fvecs'
    groundtruth_file = 'sift1M/sift_groundtruth.ivecs'
    query_file = 'sift1M/sift_query.fvecs'

    fv_base = fvecs_read(base_file)
    print("base_vectors: ", fv_base.shape)
    fv_query = fvecs_read(query_file)
    print("query_vectors: ", fv_query.shape)
    fv_groundtruth = ivecs_read(groundtruth_file)
    print("groundtruth_vectors: ", fv_groundtruth.shape)

    # test example recall rate
    # recall = calc_recall_rate(fv_groundtruth, fv_groundtruth)
    # print("recall: ", recall)

    run_tests()
