from matplotlib import pyplot as plt

nq_list = [1,3]
topk_list = [1,2]
nlist_list = [1024, 2048, 4096]
nprobe_list = [32, 64, 128]
index_type = "IVF_FLAT"

''' for different nq's '''
tpq_topk_list_dict = {
    "nq=1": {
        "1024 / 32": [0.0008999999999998899, 0.0006999999999998899],
        "2048 / 64": [0.0004999999999998899, 0.0007999999999998899],
        "4096 / 128": [0.0002999999999998899, 0.0003999999999998899],
    },
    "nq=3": {
        "1024 / 32": [0.0008999999999998899, 0.0006999999999998899],
        "2048 / 64": [0.0005999999999998899, 0.0007999999999998899],
        "4096 / 128": [0.0006999999999998899, 0.0004999999999998899],
    },
}

fig, axs = plt.subplots(nrows=1, ncols=len(nq_list), figsize= (18, 7))
fig.subplots_adjust(wspace=0.5)

for i, nq in enumerate(nq_list):
    for nlist, nprobe in zip(nlist_list, nprobe_list):
        tpq_topk_list = tpq_topk_list_dict[f"nq={nq}"]
        tpq_list = tpq_topk_list[f"{nlist} / {nprobe}"]
        print("new plot: ", i)
        axs[i].plot(topk_list, tpq_list, label=f"{nlist} / {nprobe}", marker='o')
        # add marker text
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
plt.show()


# for i, nq in enumerate(nq_list):
#     for nlist, nprobe in zip(nlist_list, nprobe_list):
#         tpq_topk_list = tpq_topk_list_dict[f"{nlist} / {nprobe}"]
#         tpq_list = []
#         for topk in topk_list:
#             tpq = tpq_topk_list[f"{nq}"]
#             tpq_list.append(tpq)
#         axs[i].plot(topk_list, tpq_list, label=f"{nlist} / {nprobe}")
#     # axs[i].xlabel('topk')
#     # axs[i].ylabel('tpq / s')
# for ax in axs.flat:
#     ax.set(xlabel='topk', ylabel='tpq / s')
# plt.title(f'TPQ vs topk for {index_type} with nq: {nq} input vectors')
# plt.legend()
# plt.show()