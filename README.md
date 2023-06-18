# Database Server Set up (Milvus)

`https://milvus.io/docs/install_standalone-docker.md`

# Dataset

I use a subset of Imagenet, called Tiny Imagenet
`https://huggingface.co/datasets/Maysee/tiny-imagenet`

# Architecture

In `Milvus/base.py` is a class (`MilvusHandler`) that provides the base functionality like creating embeddings, insert data and search data. In `Milvus/create.py` are the tests which use the `MilvusHandler` class.

# Install Embeddings
If you dont have a graphics card you can download the precomputed embeddings from here:
`pip install gdown`
Train Embeddings:
`gdown https://drive.google.com/uc?id=1-3llBB2jbYw_UsNc-0bQJnGqKyr42eHv`
Test embeddings:
`gdown https://drive.google.com/uc?id=1uFepIBTFQsGLl2ypuKRO7Ma4u0zrWULV`

# Check available memory
`df -hT`

Delete volume
`docker-compose down`
`sudo rm -rf volumes`