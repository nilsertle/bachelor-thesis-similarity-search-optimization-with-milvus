from functools import lru_cache
from datasets import load_dataset
from towhee import pipe, ops, DataCollection
import pickle
import os

class EmbeddingHandler():

    embeddings_cache_file = 'Milvus/embeddings_cache.pkl'
    test_embeddings_cache_file = 'Milvus/test_embeddings_cache.pkl'

    def __init__(self, entity_count: int=None, test_count: int=None):

        if entity_count is None:
            self.dataset = load_dataset('Maysee/tiny-imagenet', split='train')
        else:
            self.dataset = load_dataset('Maysee/tiny-imagenet', split='train').select(range(entity_count))
        if test_count is None:
            self.test_dataset = load_dataset('Maysee/tiny-imagenet', split='valid')
        else:
            self.test_dataset = load_dataset('Maysee/tiny-imagenet', split='valid').select(range(test_count))
        
        self.embeddings = None
        self.test_embeddings = None
        self.model = 'resnet50'
        self.dim = 2048
        self.embedding_model = ops.image_embedding.timm(model_name=self.model, device=None)

        self.create_embeddings()
        self.create_test_embeddings()


    def create_embedding(self, image):
        ''' 
        If no image is provided, use the first image in the dataset (for search_data) 
        '''
        
        embedding = self.embedding_model(image)
        assert len(embedding) == self.dim
        return embedding
    
    def create_embeddings(self):
        print('========== Creating embeddings... ==========')
        try:
            print(self.embeddings_cache_file)
            print(os.path.isfile(self.embeddings_cache_file))
            with open(self.embeddings_cache_file, 'rb') as f:
                cached_array = pickle.load(f)
                self.embeddings = cached_array
            print('========== Loaded embeddings from cache file... ==========')
        except Exception as e:
            print("exception: ", e)
            embeddings = []
            for i, img in enumerate(self.dataset):
                if i in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 99999]:
                    print(f"{i} embeddings created")
                embedding = self.create_embedding(img["image"])
                embeddings.append(embedding)
            self.embeddings = embeddings
            with open(self.embeddings_cache_file, 'wb') as f:
                pickle.dump(embeddings, f)
            print('========== Saved embeddings to cache file ==========')
        
        return self.embeddings
    
    def create_test_embeddings(self):
        print('========== Creating test embeddings... ==========')
        try:
            with open(self.test_embeddings_cache_file, 'rb') as f:
                cached_array = pickle.load(f)
                self.test_embeddings = cached_array
            print('========== Loaded embeddings from cache file... ==========')
        except:
            embeddings = []
            for i, img in enumerate(self.test_dataset):
                embedding = self.create_embedding(img["image"])
                embeddings.append(embedding)
            self.test_embeddings = embeddings
            with open(self.test_embeddings_cache_file, 'wb') as f:
                pickle.dump(embeddings, f)
            print('========== Saved embeddings to cache file ==========')
        
        return self.test_embeddings