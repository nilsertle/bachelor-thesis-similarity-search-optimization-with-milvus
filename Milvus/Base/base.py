from typing import Any
from pymilvus import SearchFuture, SearchResult, connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from towhee import pipe, ops, DataCollection
from datasets import load_dataset
from datasets import Image as DatasetImage
from PIL import Image
import time
from functools import lru_cache
from Base.embedding import EmbeddingHandler
import psutil
import os

class MilvusHandler:
    HOST = '127.0.0.1'
    PORT = '19530'

    def __init__(self, embedding_handler: EmbeddingHandler ,metric_type='L2', index_type='IVF_FLAT', nprobe=16, nlist=2048, topk=10, offset=0, drop_collection=True):
        self.metric_type = metric_type
        self.index_type = index_type
        self.nprobe = nprobe
        self.nlist = nlist
        self.topk = topk
        self.offset = offset

        self.collection_name = "reverse_image_search"
        self.dim = 2048
        self.drop_collection = drop_collection
        self.collection = None
        self.device = None
        self.embeddings: list[float] = embedding_handler.embeddings
        self.test_embeddings: list[float] = embedding_handler.test_embeddings
        self.test_dataset = embedding_handler.test_dataset
        self.dataset = embedding_handler.dataset
        self.embedding = None
        self.mr = None
        self.search_results = None

        connections.connect(host=self.HOST, port=self.PORT)
        self.create_milvus_collection()

        
    def create_milvus_collection(self):
        if utility.has_collection(self.collection_name) or self.drop_collection:
            print(f'Collection {self.collection_name} already exists. Dropping it...')
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

            # switch statement (self.index_type)

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
                    'params': {"nlist": self.nlist, "m": 16, "nbits": 8},
                }
            elif self.index_type in ["HNSW"]:
                index_params = {
                    'metric_type': self.metric_type,
                    'index_type': self.index_type,
                    'params': {"M": 16, "efConstruction": 500},
                }
            elif self.index_type in ["ANNOY"]:
                index_params = {
                    'metric_type': self.metric_type,
                    'index_type': self.index_type,
                    'params': {"n_trees": 8},
                }
            else:
                raise ValueError(f'Index type {self.index_type} not supported')

            self.collection.create_index(field_name='embedding', index_params=index_params)
            return self.collection
        
        self.collection = Collection(name=self.collection_name)
        return self.collection
        

    def insert_data(self):
        print('========== Inserting data... ==========')
        if len(self.embeddings) <= 0:
            raise ValueError('No embeddings to insert')
        print("hereherherherehrhre")
        print(len(self.embeddings))
        print(self.embeddings[0])
        print("hereherherherehrhre")
        # insert in batches
        batch_size = 5000
        for i in range(0, len(self.embeddings), batch_size):
            data = [
                [i for i in range(i, i+batch_size)],
                self.embeddings[i:i+batch_size],
            ]
            mr = self.collection.insert(data)
            print("mr: ", mr)

        # data = [ 
        #     [i for i in range(len(self.embeddings))],
        #     self.embeddings,
        # ]
        # mr = self.collection.insert(data)
        # print("mr: ", mr)
        # self.mr = mr
        
        # print the collection entries count
        print(f'Collection {self.collection_name} has {self.collection.num_entities} entries')

    def search_data(self, data: list[list[float]], topk=None, offset=None, metric_type=None, nprobe=None, load_collection=False):
        if len(data) <= 0:
            raise ValueError('No embeddings provided')
        
        if topk is None:
            topk = self.topk
        if offset is None:
            offset = self.offset
        if metric_type is None:
            metric_type = self.metric_type
        if nprobe is None:
            nprobe = self.nprobe
        if load_collection:
            self.collection.load()

        search_param = {
            'metric_type': metric_type,
            'params': {"nprobe": nprobe},
            'offset': offset,
        }
        results = self.collection.search(
            data=data,
            anns_field="embedding",
            param=search_param,
            limit=topk,
            expr=None,
        )
        self.search_results = results
        if load_collection:
            self.collection.release()
        return results, results[0].distances
    
    def visualize_search_results(self, results=None):
        if results is None:
            results = self.search_results

        print(results[0].ids)
        print(results[0].distances)

    def test_search(self, nq=1, topk=None, offset=None, metric_type=None, nprobe=None):
        if topk is None:
            topk = self.topk
        if offset is None:
            offset = self.offset
        if metric_type is None:
            metric_type = self.metric_type
        if nprobe is None:
            nprobe = self.nprobe

        self.collection.load()

        test_search_results = []
        print(f'========== Searching data... {self.nlist} / {nprobe} ==========')
        start_time = time.time()
        for i, te in enumerate(self.test_embeddings):
            if i in [10, 100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 9999]:
                print(f'========== Searching data... {i} ==========')
            test_search_result = self.search_data([te for x in range(nq)])
            test_search_results.append(test_search_result)
        end_time = time.time()
        print('========== Finished Searching data... ==========')

        ''' Calculate queries per second and time per query '''
        queries_per_second = (len(self.test_embeddings)) / (end_time - start_time)
        time_per_query = (end_time - start_time) / (len(self.test_embeddings))

        ''' Calculate average distance '''
        avg_distances = []
        for results in test_search_results:
            try: 
                avg_distance = 0
                for dist in results[0][0].distances:
                    avg_distance += dist
                avg_distance /= len(results[0][0].distances)
                avg_distances.append(avg_distance)
            except:
                # print("Error in average calculus: ", results)
                avg_distances.append(0)
        avg_distance = sum(avg_distances) / len(avg_distances)

        self.collection.release()

        return queries_per_second, time_per_query, avg_distance

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
        count = 500

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
