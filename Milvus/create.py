''' import base '''
from base import MilvusHandler
from embedding import EmbeddingHandler

import matplotlib.pyplot as plt

'''
Test different Indexing Algorithms for different input vector counts
'''
# INDEX_TYPES = ["FLAT", "IVF_FLAT", "IVF_SQ8", "IVF_PQ", "HNSW", "ANNOY"]
INDEX_TYPES = ["IVF_FLAT"]

def test_recall(embedding_handler: EmbeddingHandler):
    ''' PLOT: recall rate vs nlist/nprobe-pairs '''
    recall_list_dict = {}
    nlist_list = [128, 256, 512, 8192, 16384]
    nprobe_list = [4, 8, 16, 256, 512]
    assert len(nlist_list) == len(nprobe_list)
    for nlist, nprobe in zip(nlist_list, nprobe_list):
        client = MilvusHandler(embedding_handler=embedding_handler, index_type="IVF_FLAT", drop_collection=True, nlist=nlist, nprobe=nprobe)
        client.insert_data()
        recall_rate = client.test_recall_rate()
        recall_list_dict[f"{nlist} / {nprobe}"] = recall_rate

    plt.plot(recall_list_dict.keys(), recall_list_dict.values(), marker='o')
    for x, y in zip(recall_list_dict.keys(), recall_list_dict.values()):
        plt.text(x, y, f"{y:.2f}")
    plt.xlabel('nlist / nprobe')
    plt.ylabel('recall rate')
    plt.savefig(f"plots/recall_rate_vs_nlist_nprobe-pairs.png")
    plt.show()

    with open('plotdata/recall_rate_vs_nlist_nprobe-pairs.txt', 'w') as f:
        # write x in a list and y in a list to file
        f.write("x-values:\n")
        f.write(f"{list(recall_list_dict.keys())}\n")
        f.write("y-values:\n")
        f.write(f"{list(recall_list_dict.values())}\n")

    ''' PLOT: recall rate vs nq for different nlist/nprobe-pairs '''
    # nq_list = [1, 3]
    # for nlist, nprobe in zip(nlist_list, nprobe_list):
    #     client = MilvusHandler(embedding_handler=embedding_handler, index_type="IVF_FLAT", drop_collection=True, nlist=nlist, nprobe=nprobe)
    #     client.insert_data()
    #     recall_rate_list = []
    #     for nq in nq_list:
    #         recall_rate = client.test_recall_rate(nq=nq)
    #         recall_rate_list.append(recall_rate)
    #     plt.plot(nq_list, recall_rate_list, marker='o', label=f"{nlist} / {nprobe}")
    #     for x, y in zip(nq_list, recall_rate_list):
    #         plt.text(x, y, f"{y:.2f}")
    # plt.xlabel('nq')
    # plt.ylabel('recall rate')
    # plt.legend()
    # plt.savefig(f"plots/recall_rate_vs_nq.png")
    # plt.show()




def test_ivf_flat(embedding_handler: EmbeddingHandler):
    index_type = "IVF_FLAT"
    ''' nq is the number of input vectors '''

    tpq_list_dict = {}
    avg_distances_dict = {}
    nlist_list = [1024]
    nprobe_list = [32]
    tpq_topk_list_dict: dict[str, dict[str, list[float]]] = {}
    nq_list = [1,3]
    topk_list = [1,2]
    
    for nlist, nprobe in zip(nlist_list, nprobe_list):
        tpq_list = []
        avg_distances_list = []
        
        drop_collection = True
        for nq in nq_list:
            client = MilvusHandler(embedding_handler=embedding_handler, index_type=index_type, topk=10, drop_collection=drop_collection, nlist=nlist, nprobe=nprobe)
            if drop_collection:
                client.insert_data()

            qps, tpq, avg_distances = client.test_search(nq=nq)
            print("=====================================")
            print(f"nq: {nq}")
            print(f"{index_type} QPS: {qps}")
            print(f"{index_type} TPQ in ms: {tpq*1000}")
            print("=====================================")
            drop_collection = False
            tpq_list.append(tpq)
            avg_distances_list.append(avg_distances)

            ''' test different topk's '''
            little_list = []
            for topk in topk_list:
                qps, tpq, avg_distances = client.test_search(nq=nq, topk=topk)
                # tpq_topk_list[f"{nq}"] = tpq
                little_list.append(tpq)
            if f"nq={nq}" not in tpq_topk_list_dict:
                tpq_topk_list_dict[f"nq={nq}"] = {}
            tpq_topk_list_dict[f"nq={nq}"][f"{nlist} / {nprobe}"] = little_list

        tpq_list_dict[f"{nlist} / {nprobe}"] = tpq_list
        avg_distances_dict[f"{nlist} / {nprobe}"] = avg_distances_list

    ''' test ram usage '''
    client = MilvusHandler(embedding_handler=embedding_handler, index_type=index_type, topk=10, drop_collection=False)
    client.test_ram_usage()

    ''' PLOT: time per query vs input vector count '''
    for nlist, nprobe in zip(nlist_list, nprobe_list):
        tpq_list = tpq_list_dict[f"{nlist} / {nprobe}"]
        plt.plot(nq_list, tpq_list, label=f"{nlist} / {nprobe}")
    plt.xlabel('nq')
    plt.ylabel('tpq / s')
    plt.title(f'TPQ vs nq for {index_type}')
    plt.legend()
    plt.savefig(f"plots/tpq_vs_nq-{index_type}.png")
    plt.show()

    ''' PLOT: average distance vs input vector count '''
    for nlist, nprobe in zip(nlist_list, nprobe_list):
        avg_distances_list = avg_distances_dict[f"{nlist} / {nprobe}"]
        plt.plot(nq_list, avg_distances_list, label=f"{nlist} / {nprobe}")
    plt.xlabel('nq')
    plt.ylabel('avg distance')
    plt.title(f'avg distance vs nq for {index_type}')
    plt.legend()
    plt.savefig(f"plots/avg_distance_vs_nq-{index_type}.png")
    plt.show()

    ''' PLOT: time per query vs topk for different nlist / nprobe (each nq a new plot)'''
    fig, axs = plt.subplots(nrows=1, ncols=len(nq_list), figsize= (20, 8))
    fig.subplots_adjust(wspace=0.5)

    for i, nq in enumerate(nq_list):
        for nlist, nprobe in zip(nlist_list, nprobe_list):
            tpq_topk_list = tpq_topk_list_dict[f"nq={nq}"]
            tpq_list = tpq_topk_list[f"{nlist} / {nprobe}"]
            axs[i].plot(topk_list, tpq_list, label=f"{nlist} / {nprobe}", marker='o')
            for x, y in zip(topk_list, tpq_list):
                label = "{:.2f}".format(y * 1000)
                axs[i].annotate(label,
                                (x,y),
                                textcoords="offset points",
                                xytext=(0,10),
                                ha='center')
        axs[i].set(xlabel='topk', ylabel='tpq / s')
        axs[i].set_title(f'TPQ vs topk for {index_type} with nq: {nq} input vectors')
        axs[i].legend()
    plt.savefig(f"plots/tpq_vs_topk-{index_type}.png")
    plt.show()