# Database Server Set up (Milvus)

`https://milvus.io/docs/install_standalone-docker.md`

# Dataset

I use a subset of Imagenet, called Tiny Imagenet
`https://huggingface.co/datasets/Maysee/tiny-imagenet`

# Architecture

In `Milvus/base.py` is a class (`MilvusHandler`) that provides the base functionality like creating embeddings, insert data and search data. In `Milvus/create.py` are the tests which use the `MilvusHandler` class.
