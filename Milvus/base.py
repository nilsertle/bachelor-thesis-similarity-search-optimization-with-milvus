from typing import Any
from pymilvus import SearchFuture, SearchResult, connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from datasets import load_dataset
from datasets import Image as DatasetImage
from PIL import Image
import time
from functools import lru_cache
from embedding import EmbeddingHandler
import psutil
import os
import tarfile
import numpy as np
from Helpers.vecs_helper import ivecs_read, fvecs_read
from Helpers.recall_helper import calc_recall_rate

# base_file = 'siftsmall/siftsmall_base.fvecs'
# groundtruth_file = 'siftsmall/siftsmall_groundtruth.ivecs'
# query_file = 'siftsmall/siftsmall_query.fvecs'

base_file = 'sift1M/sift_base.fvecs'
groundtruth_file = 'sift1M/sift_groundtruth.ivecs'
query_file = 'sift1M/sift_query.fvecs'

class MilvusHandler:
    HOST = '127.0.0.1'
    PORT = '19530'

    nprobe = 512
    nlist = 16384
    offset = 0
    collection_name = "reverse_image_search"
    dim = 128
    m = 16
    nbits = 8
    ef_construction = 500
    ef = 500
    M = 16
    n_trees = 8

    nq = 1
    topk = 100

    def __init__(self, metric_type='L2', index_type='IVF_FLAT'):
        self.metric_type = metric_type
        self.index_type = index_type
        self.collection = None

        print("choose vectors")
        self.insert_embeddings = fvecs_read(base_file)
        # only use 1000 vectors for test
        self.query_embeddings = fvecs_read(query_file)
        self.gt = ivecs_read(groundtruth_file)
 
        connections.connect(host=self.HOST, port=self.PORT)
        self.create_milvus_collection()
        
    def create_milvus_collection(self):
        if utility.has_collection(self.collection_name) == False:
            print(f'Collection {self.collection_name} does not exist or drop_collection was set to True. Dropping it...')
            utility.drop_collection(self.collection_name)

            fields = [
                FieldSchema(
                    name="img_id",
                    dtype=DataType.INT64,
                    is_primary=True,
                ),
                FieldSchema(
                    name='embedding',
                    dtype=DataType.FLOAT_VECTOR,
                    description='image embedding vectors',
                    dim=self.dim,
                    auto_id=False,
                ),
            ]
            schema = CollectionSchema(fields=fields, description='reverse image search')
            self.collection = Collection(name=self.collection_name, schema=schema)

            if self.index_type in ["IVF_FLAT", "IVF_SQ8", "FLAT"]:
                index_params = {
                    'metric_type': self.metric_type,
                    'index_type': self.index_type,
                    'params': {"nlist": self.nlist},
                }
            elif self.index_type in ["IVF_PQ"]:
                index_params = {
                    'metric_type': self.metric_type,
                    'index_type': self.index_type,
                    'params': {"nlist": self.nlist, "m": self.m, "nbits": self.nbits},
                }
            elif self.index_type in ["HNSW"]:
                index_params = {
                    'metric_type': self.metric_type,
                    'index_type': self.index_type,
                    'params': {"M": self.M, "efConstruction": self.ef_construction},
                }
            elif self.index_type in ["ANNOY"]:
                index_params = {
                    'metric_type': self.metric_type,
                    'index_type': self.index_type,
                    'params': {"n_trees": self.n_trees},
                }
            else:
                raise ValueError(f'Index type {self.index_type} not supported')

            self.collection.create_index(field_name='embedding', index_params=index_params)
        else:
            self.collection = Collection(name=self.collection_name)

        

        print(f'========== Entities in DB: {self.collection.num_entities} ==========')
        if self.collection.num_entities == 0:
            self.insert_data()
        return self.collection

    def insert_data(self):
        print(f'========== Inserting data... embedding count: {len(self.insert_embeddings)} ==========')
        if len(self.insert_embeddings) <= 0:
            raise ValueError('No embeddings to insert')
       
        batch_size = 8000 
        total_embeddings = len(self.insert_embeddings)
        num_batches = (total_embeddings + batch_size - 1) // batch_size
        
        for batch_index in range(num_batches):
            print(f'Inserting batch {batch_index + 1} of {num_batches}')
            start_index = batch_index * batch_size
            end_index = min((batch_index + 1) * batch_size, total_embeddings)
            
            batch_data = [
                [i for i in range(start_index, end_index)],
                self.insert_embeddings[start_index:end_index],
            ]
            
            self.collection.insert(batch_data)



    def search_data(self, data):
        if self.index_type in ["IVF_FLAT", "IVF_SQ8", "FLAT"]:
            search_param = {
                'metric_type': self.metric_type,
                'params': {"nprobe": self.nprobe},
                'offset': self.offset,
            }
        elif self.index_type in ["IVF_PQ"]:
            search_param = {
                'metric_type': self.metric_type,
                'params': {"nprobe": self.nprobe},
                'offset': self.offset,
            }
        elif self.index_type in ["HNSW"]:
            search_param = {
                'metric_type': self.metric_type,
                'params': {"ef": self.ef},
                'offset': self.offset,
            }
        elif self.index_type in ["ANNOY"]:
            search_param = {
                'metric_type': self.metric_type,
                'params': {"search_k": self.ef},
                'offset': self.offset,
            }
        else:
            raise ValueError(f'Index type {self.index_type} not supported')
        
        results = self.collection.search(
            data=data,
            anns_field="embedding",
            param=search_param,
            limit=self.topk,
            expr=None,
        )
        self.search_results = results
        return results

    def test_search(self):
        print(f'========== Loading data into memory... ==========')

        self.collection.load()

        test_search_results = []
        print(f'========== Searching data... {self.nlist} / {self.nprobe} ==========')
        start_time = time.time()
        for te in self.query_embeddings:
            test_search_result = self.search_data([te])
            test_search_results.append(test_search_result)
        end_time = time.time()
        print('========== Finished Searching data... ==========')

        pid = os.getpid()
        process = psutil.Process(pid)   
        ram_usage = process.memory_info().rss / 1024 ** 2
        print(f'RAM usage: {ram_usage} MB')

        self.collection.release()

        ''' Calculate queries per second and time per query '''
        queries_per_second = (len(self.query_embeddings)) / (end_time - start_time)
        time_per_query = (end_time - start_time) / (len(self.query_embeddings))

        ''' Calculate recall rate with calc_recall_rate() '''
        results = np.array([x[0].ids for x in test_search_results])

        print("shape of results: ", results.shape)
        print("shape of groundtruth_vectors: ", self.gt.shape)
        print("results: ", results)
        print("are same? ", np.array_equal(results, self.gt))

        recall = calc_recall_rate(self.gt, results)

        return queries_per_second, time_per_query, recall

    def test_ram_usage(self):
        self.collection.load()
        pid = os.getpid()
        process = psutil.Process(pid)   
        ram_usage = process.memory_info().rss / 1024 ** 2
        self.collection.release()
        print(f'RAM usage: {ram_usage} MB')
        return ram_usage

    def test_recall_rate(self, nq=1, topk=None):
        # test topk = len(images per category)
        self.collection.load()

        # count should be len of images per category (500)
        count = 100

        print(f'========== Searching data... {self.nlist} / {self.nprobe} ==========')
        test_search_results: list[tuple[SearchResult, Any | list]] = []
        for te in self.test_embeddings:
            test_search_result = self.search_data([te for x in range(nq)], topk=count)
            test_search_results.append(test_search_result)
        
        recall_rates = []
        for i, results in enumerate(test_search_results):
            recall_rate = 0
            for id in results[0][0].ids:
                if self.test_dataset[i]["label"] == self.dataset[id]["label"]:
                    recall_rate += 1
            recall_rate /= count
            recall_rates.append(recall_rate)
        average_recall_rate = sum(recall_rates) / len(recall_rates)

        self.collection.release()

        return average_recall_rate
