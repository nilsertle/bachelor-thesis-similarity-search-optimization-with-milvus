# import matlab plotlib
import matplotlib.pyplot as plt
import numpy as np

'''
nlist = 16384
FLAT, IVF_FLAT, IVF_SQ8
'''

nprobe = np.array([2, 8, 32, 128, 512])

qps_FLAT = np.array([16.23, 16.13, 16.74, 16.86, 17.72])
qps_IVF_FLAT = np.array([11.29, 11.96, 7.78, 10.68, 6.564])
qps_IVF_SQ8 = np.array([7.43, 10.22, 12.50, 12.65, 8.33])

recalls_FLAT = np.array([[0.9929, 1.0, 1.0], [0.9925, 1.0, 1.0], [0.9928, 1.0, 1.0], [0.9936, 1.0, 1.0], [0.9931, 1.0, 1.0]])
recalls_IVF_FLAT = np.array([[0.908, 0.912, 0.912], [0.9133, 0.9191, 0.9191], [0.9711, 0.9789, 0.9789], [0.9907, 0.9991, 0.9991], [0.992, 1.0, 1.0]])
recalls_IVF_SQ8 = np.array([[0.9018, 0.9076, 0.9076], [0.9154, 0.9256, 0.9256], [0.9787, 0.989, 0.989], [0.9821, 0.9978, 0.9978], [0.9843, 1.0, 1.0]])

plt.figure()
plt.plot(nprobe, qps_FLAT, label='FLAT', marker='o')
plt.plot(nprobe, qps_IVF_FLAT, label='IVF_FLAT', marker='o')
plt.plot(nprobe, qps_IVF_SQ8, label='IVF_SQ8', marker='o')
plt.xlabel('nprobe')
plt.ylabel('qps')
plt.title('nprobe vs qps for nlist=16384')
plt.legend()
plt.savefig('nprobe_vs_qps.png')
plt.close()

plt.figure()
plt.plot(nprobe, recalls_FLAT[:, 0], label='FLAT', marker='o')
plt.plot(nprobe, recalls_IVF_FLAT[:, 0], label='IVF_FLAT', marker='o')
plt.plot(nprobe, recalls_IVF_SQ8[:, 0], label='IVF_SQ8', marker='o')
plt.xlabel('nprobe')
plt.ylabel('recall')
plt.title('nprobe vs recall@1 for nlist=16384')
plt.legend()
plt.savefig('nprobe_vs_recall.png')
plt.close()

'''
IVF_PQ
nbits = 8
'''

qps_m2 = np.array([7.58, 7.44, 10.52, 11.79, 13.57])
qps_m8 = np.array([12.45, 11.41, 10.97, 12.59, 12.40])
qps_m32 = np.array([11.5, 9.89, 12.12, 12.13, 11.74])

recalls_m2 = np.array([[0.5618, 0.8508, 0.906], [0.5501, 0.8602, 0.9543], [0.5412, 0.8554, 0.9736], [0.5424, 0.849, 0.9771], [0.3926, 0.7247, 0.941]])
recalls_m8 = np.array([[0.7669, 0.905, 0.9129], [0.7515, 0.9413, 0.96], [0.7397, 0.9568, 0.988], [0.6268, 0.9291, 0.9958], [0.6168, 0.9296, 0.9959]])
recalls_m32 = np.array([[0.8746, 0.9156, 0.9156], [0.893, 0.9607, 0.9612], [0.9026, 0.9887, 0.99], [0.9089, 0.9977, 0.9992], [0.9134, 0.9988, 1.0]])

plt.figure()
plt.plot(nprobe, qps_m2, label='m=2', marker='o')
plt.plot(nprobe, qps_m8, label='m=8', marker='o')
plt.plot(nprobe, qps_m32, label='m=32', marker='o')
plt.xlabel('nprobe')
plt.ylabel('qps')
plt.title('IVF_PQ nprobe vs qps for nlist=16384 and nbits=8')
plt.legend()
plt.savefig('nprobe_vs_qps_IVF_PQ.png')
plt.close()

plt.figure()
plt.plot(nprobe, recalls_m2[:, 0], label='m=2', marker='o')
plt.plot(nprobe, recalls_m8[:, 0], label='m=8', marker='o')
plt.plot(nprobe, recalls_m32[:, 0], label='m=32', marker='o')
plt.xlabel('nprobe')
plt.ylabel('recall')
plt.title('IVF_PQ nprobe vs recall@1 for nlist=16384 and nbits=8')
plt.legend()
plt.savefig('nprobe_vs_recall_IVF_PQ.png')
plt.close()

'''
HNSW
'''

ef = np.array([128, 512, 2048, 8192])

qps_M8_efconstruction8 = np.array([66.46, 75.52, 62.05, 56.17])
qps_M16_efconstruction16 = np.array([62.24, 67.95, 57.23, 47.69])
qps_M32_efconstruction32 = np.array([60.14, 70.18, 58.33, 39.07])
qps_M64_efconstruction64 = np.array([47.46, 39.00, 47.41, 30.16])
qps_M64_efconstruction128 = np.array([32.15, 37.42, 29.31, 21.73])
qps_M64_efconstruction256 = np.array([23.29, 23.84, 22.85, 14.68])
qps_M64_efconstruction512 = np.array([15.11, 9.73, 9.14, 6.20])

recalls_M8_efconstruction8 = np.array([[0.6191, 0.6213, 0.6213], [0.7909, 0.7959, 0.7959], [0.8222, 0.8284, 0.8284], [0.9306, 0.9365, 0.9365]])
recalls_M16_efconstruction16 = np.array([[0.9261, 0.9334, 0.9334], [0.9729, 0.98, 0.98], [0.9748, 0.9819, 0.9819], [0.9617, 0.9689, 0.9689]])
recalls_M32_efconstruction32 = np.array([[0.9834, 0.9903, 0.9903], [0.9889, 0.996, 0.996], [0.993, 0.9998, 0.9998], [0.9926, 1.0, 1.0]])
recalls_M64_efconstruction64 = np.array([[0.989, 0.9967, 0.9967], [0.9916, 0.9997, 0.9997], [0.9925, 1.0, 1.0], [0.9922, 0.9999, 0.9999]])
recalls_M64_efconstruction128 = np.array([[0.9906, 0.9992, 0.9992], [0.9923, 1.0, 1.0], [0.9927, 1.0, 1.0], [0.993, 0.9999, 0.9999]])
recalls_M64_efconstruction256 = np.array([[0.9913, 0.9995, 0.9995], [0.9924, 1.0, 1.0], [0.9922, 1.0, 1.0], [0.9925, 1.0, 1.0]])
recalls_M64_efconstruction512 = np.array([[0.9914, 0.9998, 0.9998], [0.9913, 1.0, 1.0], [0.9915, 1.0, 1.0], [0.9922, 1.0, 1.0]])

plt.figure()
plt.plot(ef, qps_M8_efconstruction8, label='M=8 efConstruction=8', marker='o')
plt.plot(ef, qps_M16_efconstruction16, label='M=16 efConstruction=16', marker='o')
plt.plot(ef, qps_M32_efconstruction32, label='M=32 efConstruction=32', marker='o')
plt.plot(ef, qps_M64_efconstruction64, label='M=64 efConstruction=64', marker='o')
plt.plot(ef, qps_M64_efconstruction128, label='M=64 efConstruction=128', marker='o')
plt.plot(ef, qps_M64_efconstruction256, label='M=64 efConstruction=256', marker='o')
plt.plot(ef, qps_M64_efconstruction512, label='M=64 efConstruction=512', marker='o')
plt.xlabel('ef')
plt.ylabel('qps')
plt.title('HNSW ef vs qps')
plt.legend()
plt.savefig('ef_vs_qps_HNSW.png')
plt.close()

plt.figure()
plt.plot(ef, recalls_M8_efconstruction8[:, 0], label='M=8 efConstruction=8', marker='o')
plt.plot(ef, recalls_M16_efconstruction16[:, 0], label='M=16 efConstruction=16', marker='o')
plt.plot(ef, recalls_M32_efconstruction32[:, 0], label='M=32 efConstruction=32', marker='o')
plt.plot(ef, recalls_M64_efconstruction64[:, 0], label='M=64 efConstruction=64', marker='o')
plt.plot(ef, recalls_M64_efconstruction128[:, 0], label='M=64 efConstruction=128', marker='o')
plt.plot(ef, recalls_M64_efconstruction256[:, 0], label='M=64 efConstruction=256', marker='o')
plt.plot(ef, recalls_M64_efconstruction512[:, 0], label='M=64 efConstruction=512', marker='o')
plt.xlabel('ef')
plt.ylabel('recall')
plt.title('HNSW ef vs recall@1')
plt.legend()
plt.savefig('ef_vs_recall_HNSW.png')
plt.close()

'''
ANNOY
'''

search_k = np.array([512, 8192, 131072, 2097152])

qps_ntrees4 = np.array([41.49, 34.26, 25.44, 4.02])
qps_ntrees16 = np.array([43.44, 59.27, 26.85, 4.06])
qps_ntrees64 = np.array([25.69, 24.17, 21.54, 4.51])
qps_ntrees256 = np.array([17.11, 14.79, 16.57 ])

recalls_ntrees4 = np.array([[0.644, 0.6484, 0.6484], [0.8952, 0.902, 0.902], [0.9433, 0.9499, 0.9499], [0.9828, 0.9914, 0.9914]])
recalls_ntrees16 = np.array([[0.691, 0.6949, 0.6949], [0.9509, 0.9583, 0.9583], [0.9406, 0.9472, 0.9472], [0.9866, 0.9926, 0.9926]])
recalls_ntrees64 = np.array([[0.5569, 0.5616, 0.5616], [0.6789, 0.6856, 0.6856], [0.7137, 0.7204, 0.7204], [0.9354, 0.9431, 0.9431]])
recalls_ntrees256 = np.array([[0.3866, 0.389, 0.389], [0.4361, 0.4385, 0.4385], [0.4288, 0.4314, 0.4314]])

plt.figure()
plt.plot(search_k, qps_ntrees4, label='ntrees=4', marker='o')
plt.plot(search_k, qps_ntrees16, label='ntrees=16', marker='o')
plt.plot(search_k, qps_ntrees64, label='ntrees=64', marker='o')
plt.plot(search_k[:3], qps_ntrees256, label='ntrees=256', marker='o')
plt.xlabel('search_k')
plt.ylabel('qps')
plt.title('ANNOY search_k vs qps')
plt.legend()
plt.savefig('search_k_vs_qps_ANNOY.png')
plt.close()

plt.figure()
plt.plot(search_k, recalls_ntrees4[:, 0], label='ntrees=4', marker='o')
plt.plot(search_k, recalls_ntrees16[:, 0], label='ntrees=16', marker='o')
plt.plot(search_k, recalls_ntrees64[:, 0], label='ntrees=64', marker='o')
plt.plot(search_k[:3], recalls_ntrees256[:, 0], label='ntrees=256', marker='o')
plt.xlabel('search_k')
plt.ylabel('recall')
plt.title('ANNOY search_k vs recall@1')
plt.legend()
plt.savefig('search_k_vs_recall_ANNOY.png')
plt.close()

'''
nq plots with recall@1 and qps
'''

nq = np.array([1, 10, 100])

# FLAT
qps_FLAT_nprobe8 = np.array([18.07, 6.91])
qps_FLAT_nprobe64 = np.array([18.04, 7.43])

recalls_FLAT_nprobe8 = np.array([[0.9927, 1.0, 1.0], [0.9924, 1.0, 1.0]])
recalls_FLAT_nprobe64 = np.array([[0.9925, 1.0, 1.0], [0.9921, 1.0, 1.0]])

plt.figure()
plt.plot(nq[:2], qps_FLAT_nprobe8, label='nprobe=8', marker='o')
plt.plot(nq[:2], qps_FLAT_nprobe64, label='nprobe=64', marker='o')
plt.xlabel('nq')
plt.ylabel('qps')
plt.title('FLAT nq vs qps')
plt.legend()
plt.savefig('nq_vs_qps_FLAT.png')
plt.close()

plt.figure()
plt.plot(nq[:2], recalls_FLAT_nprobe8[:, 0], label='nprobe=8', marker='o')
plt.plot(nq[:2], recalls_FLAT_nprobe64[:, 0], label='nprobe=64', marker='o')
plt.xlabel('nq')
plt.ylabel('recall')
plt.title('FLAT nq vs recall@1')
plt.legend()
plt.savefig('nq_vs_recall_FLAT.png')
plt.close()

# IVF_FLAT
qps_IVF_FLAT_nprobe8 = np.array([12.93, 6.89, 3.55])
qps_IVF_FLAT_nprobe64 = np.array([13.36, 6.33, 3.75])

recalls_IVF_FLAT_nprobe8 = np.array([[0.9169, 0.9227, 0.9227], [0.8059, 0.8123, 0.8123], [0.7271, 0.7327, 0.7327]])
recalls_IVF_FLAT_nprobe64 = np.array([[0.986, 0.9943, 0.9943], [0.9707, 0.979, 0.979], [0.9543, 0.9617, 0.9617]])

plt.figure()
plt.plot(nq, qps_IVF_FLAT_nprobe8, label='nprobe=8', marker='o')
plt.plot(nq, qps_IVF_FLAT_nprobe64, label='nprobe=64', marker='o')
plt.xlabel('nq')
plt.ylabel('qps')
plt.title('IVF_FLAT nq vs qps')
plt.legend()
plt.savefig('nq_vs_qps_IVF_FLAT.png')
plt.close()

plt.figure()
plt.plot(nq, recalls_IVF_FLAT_nprobe8[:, 0], label='nprobe=8', marker='o')
plt.plot(nq, recalls_IVF_FLAT_nprobe64[:, 0], label='nprobe=64', marker='o')
plt.xlabel('nq')
plt.ylabel('recall')
plt.title('IVF_FLAT nq vs recall@1')
plt.legend()
plt.savefig('nq_vs_recall_IVF_FLAT.png')
plt.close()

# IVF_SQ8
qps_IVF_SQ8_nprobe8 = np.array([11.12, 5.87, 3.51])
qps_IVF_SQ8_nprobe64 = np.array([13.37, 6.23, 3.82])

recalls_IVF_SQ8_nprobe8 = np.array([[0.9512, 0.9599, 0.9599], [0.8087, 0.8206, 0.8206], [0.7151, 0.7265, 0.7265]])
recalls_IVF_SQ8_nprobe64 = np.array([[0.9803, 0.9943, 0.9943], [0.9558, 0.9759, 0.9759], [0.9395, 0.96, 0.96]])

plt.figure()
plt.plot(nq, qps_IVF_SQ8_nprobe8, label='nprobe=8', marker='o')
plt.plot(nq, qps_IVF_SQ8_nprobe64, label='nprobe=64', marker='o')
plt.xlabel('nq')
plt.ylabel('qps')
plt.title('IVF_SQ8 nq vs qps')
plt.legend()
plt.savefig('nq_vs_qps_IVF_SQ8.png')
plt.close()

plt.figure()
plt.plot(nq, recalls_IVF_SQ8_nprobe8[:, 0], label='nprobe=8', marker='o')
plt.plot(nq, recalls_IVF_SQ8_nprobe64[:, 0], label='nprobe=64', marker='o')
plt.xlabel('nq')
plt.ylabel('recall')
plt.title('IVF_SQ8 nq vs recall@1')
plt.legend()
plt.savefig('nq_vs_recall_IVF_SQ8.png')
plt.close()

# IVF_PQ
qps_IVF_PQ_nprobe8 = np.array([11.40, 6.09, 3.67])
qps_IVF_PQ_nprobe64 = np.array([11.26, 5.51, 3.52])

recalls_IVF_PQ_nprobe8 = np.array([[0.8382, 0.9562, 0.9616], [0.5771, 0.7987, 0.8103], [0.4536, 0.7126, 0.7243]])
recalls_IVF_PQ_nprobe64 = np.array([[0.8506, 0.986, 0.9968], [0.6335, 0.9431, 0.9754], [0.5239, 0.9156, 0.961]])

plt.figure()
plt.plot(nq, qps_IVF_PQ_nprobe8, label='nprobe=8', marker='o')
plt.plot(nq, qps_IVF_PQ_nprobe64, label='nprobe=64', marker='o')
plt.xlabel('nq')
plt.ylabel('qps')
plt.title('IVF_PQ nq vs qps')
plt.legend()
plt.savefig('nq_vs_qps_IVF_PQ.png')
plt.close()

plt.figure()
plt.plot(nq, recalls_IVF_PQ_nprobe8[:, 0], label='nprobe=8', marker='o')
plt.plot(nq, recalls_IVF_PQ_nprobe64[:, 0], label='nprobe=64', marker='o')
plt.xlabel('nq')
plt.ylabel('recall')
plt.title('IVF_PQ nq vs recall@1')
plt.legend()
plt.savefig('nq_vs_recall_IVF_PQ.png')
plt.close()

# HNSW
qps_HNSW_nprobe8 = np.array([46.40, 25.84, 10.97])
qps_HNSW_nprobe64 = np.array([47.86, 19.19, 6.38])

recalls_HNSW_nprobe8 = np.array([[0.9891, 0.9963, 0.9963], [0.9852, 0.9937, 0.9937], [0.9873, 0.9932, 0.9932]])
recalls_HNSW_nprobe64 = np.array([[0.9921, 0.9998, 0.9998], [0.9915, 0.9999, 0.9999], [0.9911, 0.9997, 0.9997]])

plt.figure()
plt.plot(nq, qps_HNSW_nprobe8, label='nprobe=8', marker='o')
plt.plot(nq, qps_HNSW_nprobe64, label='nprobe=64', marker='o')
plt.xlabel('nq')
plt.ylabel('qps')
plt.title('HNSW nq vs qps')
plt.legend()
plt.savefig('nq_vs_qps_HNSW.png')
plt.close()

plt.figure()
plt.plot(nq, recalls_HNSW_nprobe8[:, 0], label='nprobe=8', marker='o')
plt.plot(nq, recalls_HNSW_nprobe64[:, 0], label='nprobe=64', marker='o')
plt.xlabel('nq')
plt.ylabel('recall')
plt.title('HNSW nq vs recall@1')
plt.legend()
plt.savefig('nq_vs_recall_HNSW.png')
plt.close()

# ANNOY
qps_ANNOY_nprobe8 = np.array([55.47, 36.17, 11.66])
qps_ANNOY_nprobe64 = np.array([61.09, 28.71, 9.59])

recalls_ANNOY_nprobe8 = np.array([[0.7405, 0.7453, 0.7453], [0.7191, 0.7232, 0.7232], [0.6823, 0.6864, 0.6864]])
recalls_ANNOY_nprobe64 = np.array([[0.9386, 0.945, 0.945], [0.929, 0.9363, 0.9363], [0.9565, 0.9646, 0.9646]])

plt.figure()
plt.plot(nq, qps_ANNOY_nprobe8, label='nprobe=8', marker='o')
plt.plot(nq, qps_ANNOY_nprobe64, label='nprobe=64', marker='o')
plt.xlabel('nq')
plt.ylabel('qps')
plt.title('ANNOY nq vs qps')
plt.legend()
plt.savefig('nq_vs_qps_ANNOY.png')
plt.close()

plt.figure()
plt.plot(nq, recalls_ANNOY_nprobe8[:, 0], label='nprobe=8', marker='o')
plt.plot(nq, recalls_ANNOY_nprobe64[:, 0], label='nprobe=64', marker='o')
plt.xlabel('nq')
plt.ylabel('recall')
plt.title('ANNOY nq vs recall@1')
plt.legend()
plt.savefig('nq_vs_recall_ANNOY.png')
plt.close()

'''
qps vs recall@1
'''

plt.figure(figsize=(18, 10))
plt.plot(recalls_FLAT[:, 0], qps_FLAT, label='FLAT', marker='o')
plt.plot(recalls_IVF_FLAT[:, 0], qps_IVF_FLAT, label='IVF_FLAT', marker='o')
plt.plot(recalls_IVF_SQ8[:, 0], qps_IVF_SQ8, label='IVF_SQ8', marker='o')
plt.plot(recalls_m2[:, 0], qps_m2, label='IVF_PQ m2', marker='o')
plt.plot(recalls_m8[:, 0], qps_m8, label='IVF_PQ m8', marker='o')
plt.plot(recalls_m32[:, 0], qps_m32, label='IVF_PQ m32', marker='o')
plt.plot(recalls_M8_efconstruction8[:, 0], qps_M8_efconstruction8, label='HNSW M8_efconstruction8', marker='o')
plt.plot(recalls_M16_efconstruction16[:, 0], qps_M16_efconstruction16, label='HNSW M16_efconstruction16', marker='o')

# ==================== best trade off ====================
plt.plot(recalls_M32_efconstruction32[:, 0], qps_M32_efconstruction32, label='HNSW M32_efconstruction32', marker='o')
for i, txt in enumerate(zip(qps_M32_efconstruction32, recalls_M32_efconstruction32[:, 0])):
    plt.annotate(txt, (recalls_M32_efconstruction32[i, 0], qps_M32_efconstruction32[i]), fontsize=10)

plt.plot(recalls_M64_efconstruction64[:, 0], qps_M64_efconstruction64, label='HNSW M64_efconstruction64', marker='o')
plt.plot(recalls_M64_efconstruction128[:, 0], qps_M64_efconstruction128, label='HNSW M64_efconstruction128', marker='o')
plt.plot(recalls_M64_efconstruction256[:, 0], qps_M64_efconstruction256, label='HNSW M64_efconstruction256', marker='o')
plt.plot(recalls_M64_efconstruction512[:, 0], qps_M64_efconstruction512, label='HNSW M64_efconstruction512', marker='o')
plt.plot(recalls_ntrees4[:, 0], qps_ntrees4, label='ANNOY ntrees=4', marker='o')
plt.plot(recalls_ntrees16[:, 0], qps_ntrees16, label='ANNOY ntrees=16', marker='o')
plt.plot(recalls_ntrees64[:, 0], qps_ntrees64, label='ANNOY ntrees=64', marker='o')
plt.plot(recalls_ntrees256[:, 0], qps_ntrees256, label='ANNOY ntrees=256', marker='o')

plt.xscale('logit')
plt.xlabel('recall@1')
plt.ylabel('qps')
plt.title('qps vs recall@1')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.savefig('qps_vs_recall.png', bbox_inches='tight')
plt.close()


'''
closer view for the best choice
'''

plt.figure(figsize=(18, 10))
plt.plot(recalls_FLAT[:, 0], qps_FLAT, label='FLAT', marker='o')
plt.plot(recalls_IVF_FLAT[:, 0], qps_IVF_FLAT, label='IVF_FLAT', marker='o')
plt.plot(recalls_IVF_SQ8[:, 0], qps_IVF_SQ8, label='IVF_SQ8', marker='o')
# plt.plot(recalls_m2[:, 0], qps_m2, label='IVF_PQ m2', marker='o') # accuracy too low
# plt.plot(recalls_m8[:, 0], qps_m8, label='IVF_PQ m8', marker='o')
plt.plot(recalls_m32[:, 0], qps_m32, label='IVF_PQ m32', marker='o')
plt.plot(recalls_M8_efconstruction8[-2:, 0], qps_M8_efconstruction8[-2:], label='HNSW M8_efconstruction8', marker='o')
plt.plot(recalls_M16_efconstruction16[:, 0], qps_M16_efconstruction16, label='HNSW M16_efconstruction16', marker='o')

# ==================== best trade off ====================
plt.plot(recalls_M32_efconstruction32[:, 0], qps_M32_efconstruction32, label='HNSW M32_efconstruction32', marker='o')
for i, txt in enumerate(zip(qps_M32_efconstruction32, recalls_M32_efconstruction32[:, 0])):
    plt.annotate(txt, (recalls_M32_efconstruction32[i, 0], qps_M32_efconstruction32[i]), fontsize=10)

plt.plot(recalls_M64_efconstruction64[:, 0], qps_M64_efconstruction64, label='HNSW M64_efconstruction64', marker='o')
plt.plot(recalls_M64_efconstruction128[:, 0], qps_M64_efconstruction128, label='HNSW M64_efconstruction128', marker='o')
plt.plot(recalls_M64_efconstruction256[:, 0], qps_M64_efconstruction256, label='HNSW M64_efconstruction256', marker='o')
plt.plot(recalls_M64_efconstruction512[:, 0], qps_M64_efconstruction512, label='HNSW M64_efconstruction512', marker='o')
plt.plot(recalls_ntrees4[-3:, 0], qps_ntrees4[-3:], label='ANNOY ntrees=4', marker='o')
plt.plot(recalls_ntrees16[-3:, 0], qps_ntrees16[-3:], label='ANNOY ntrees=16', marker='o')
# plt.plot(recalls_ntrees64[:, 0], qps_ntrees64, label='ANNOY ntrees=64', marker='o') # accuracy too low
# plt.plot(recalls_ntrees256[:, 0], qps_ntrees256, label='ANNOY ntrees=256', marker='o') # accuracy too low
# use 1/logarithmic scale for x axis
plt.xscale('logit')
plt.xlabel('recall@1')
plt.ylabel('qps')
plt.title('qps vs recall@1')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.savefig('qps_vs_recall_closeup.png', bbox_inches='tight')
plt.close()


'''
qps vs recall@1 for different nqs
'''

# nq = 1
plt.figure(figsize=(18, 10))
# qps_FLAT_nprobe8 = np.array([18.07, 6.91])
# qps_FLAT_nprobe64 = np.array([18.04, 7.43])

# recalls_FLAT_nprobe8 = np.array([[0.9927, 1.0, 1.0], [0.9924, 1.0, 1.0]])
# recalls_FLAT_nprobe64 = np.array([[0.9925, 1.0, 1.0], [0.9921, 1.0, 1.0]])

plt.plot(recalls_FLAT_nprobe8[0, 0], qps_FLAT_nprobe8[0], label='FLAT nprobe=8', marker='o')
plt.plot(recalls_FLAT_nprobe64[0, 0], qps_FLAT_nprobe64[0], label='FLAT nprobe=64', marker='o')
plt.plot(recalls_IVF_FLAT_nprobe8[0, 0], qps_IVF_FLAT_nprobe8[0], label='IVF_FLAT nprobe=8', marker='o')
plt.plot(recalls_IVF_FLAT_nprobe64[0, 0], qps_IVF_FLAT_nprobe64[0], label='IVF_FLAT nprobe=64', marker='o')
plt.plot(recalls_IVF_SQ8_nprobe8[0, 0], qps_IVF_SQ8_nprobe8[0], label='IVF_SQ8 nprobe=8', marker='o')
plt.plot(recalls_IVF_SQ8_nprobe64[0, 0], qps_IVF_SQ8_nprobe64[0], label='IVF_SQ8 nprobe=64', marker='o')
plt.plot(recalls_IVF_PQ_nprobe8[0, 0], qps_IVF_PQ_nprobe8[0], label='IVF_PQ nprobe=8', marker='o')
plt.plot(recalls_IVF_PQ_nprobe64[0, 0], qps_IVF_PQ_nprobe64[0], label='IVF_PQ nprobe=64', marker='o')
plt.plot(recalls_HNSW_nprobe8[0, 0], qps_HNSW_nprobe8[0], label='HNSW nprobe=8', marker='o')
plt.plot(recalls_HNSW_nprobe64[0, 0], qps_HNSW_nprobe64[0], label='HNSW nprobe=64', marker='o')
plt.plot(recalls_ANNOY_nprobe8[0, 0], qps_ANNOY_nprobe8[0], label='ANNOY nprobe=8', marker='o')
plt.plot(recalls_ANNOY_nprobe64[0, 0], qps_ANNOY_nprobe64[0], label='ANNOY nprobe=64', marker='o')

# recalls_FLAT_nprobe8[0, 0] = 0.9927
# write values at each point in format (qps, recall@1)
plt.text(recalls_FLAT_nprobe8[0, 0], qps_FLAT_nprobe8[0], '({:.4f}, {:.4f})'.format(recalls_FLAT_nprobe8[0, 0], qps_FLAT_nprobe8[0]))
plt.text(recalls_FLAT_nprobe64[0, 0], qps_FLAT_nprobe64[0], '({:.4f}, {:.4f})'.format(recalls_FLAT_nprobe64[0, 0], qps_FLAT_nprobe64[0]))
plt.text(recalls_IVF_FLAT_nprobe8[0, 0], qps_IVF_FLAT_nprobe8[0], '({:.4f}, {:.4f})'.format(recalls_IVF_FLAT_nprobe8[0, 0], qps_IVF_FLAT_nprobe8[0]))
plt.text(recalls_IVF_FLAT_nprobe64[0, 0], qps_IVF_FLAT_nprobe64[0], '({:.4f}, {:.4f})'.format(recalls_IVF_FLAT_nprobe64[0, 0], qps_IVF_FLAT_nprobe64[0]))
plt.text(recalls_IVF_SQ8_nprobe8[0, 0], qps_IVF_SQ8_nprobe8[0], '({:.4f}, {:.4f})'.format(recalls_IVF_SQ8_nprobe8[0, 0], qps_IVF_SQ8_nprobe8[0]))
plt.text(recalls_IVF_SQ8_nprobe64[0, 0], qps_IVF_SQ8_nprobe64[0], '({:.4f}, {:.4f})'.format(recalls_IVF_SQ8_nprobe64[0, 0], qps_IVF_SQ8_nprobe64[0]))
plt.text(recalls_IVF_PQ_nprobe8[0, 0], qps_IVF_PQ_nprobe8[0], '({:.4f}, {:.4f})'.format(recalls_IVF_PQ_nprobe8[0, 0], qps_IVF_PQ_nprobe8[0]))
plt.text(recalls_IVF_PQ_nprobe64[0, 0], qps_IVF_PQ_nprobe64[0], '({:.4f}, {:.4f})'.format(recalls_IVF_PQ_nprobe64[0, 0], qps_IVF_PQ_nprobe64[0]))
plt.text(recalls_HNSW_nprobe8[0, 0], qps_HNSW_nprobe8[0], '({:.4f}, {:.4f})'.format(recalls_HNSW_nprobe8[0, 0], qps_HNSW_nprobe8[0]))
plt.text(recalls_HNSW_nprobe64[0, 0], qps_HNSW_nprobe64[0], '({:.4f}, {:.4f})'.format(recalls_HNSW_nprobe64[0, 0], qps_HNSW_nprobe64[0]))
plt.text(recalls_ANNOY_nprobe8[0, 0], qps_ANNOY_nprobe8[0], '({:.4f}, {:.4f})'.format(recalls_ANNOY_nprobe8[0, 0], qps_ANNOY_nprobe8[0]))
plt.text(recalls_ANNOY_nprobe64[0, 0], qps_ANNOY_nprobe64[0], '({:.4f}, {:.4f})'.format(recalls_ANNOY_nprobe64[0, 0], qps_ANNOY_nprobe64[0]))


plt.xlabel('recall@1')
plt.ylabel('qps')
plt.title('qps vs recall@1 for nq=1')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.savefig('qps_vs_recall_nq1.png', bbox_inches='tight')
plt.close()

# nq = 10
plt.figure(figsize=(18, 10))
plt.plot(recalls_FLAT_nprobe8[1, 0], qps_FLAT_nprobe8[1], label='FLAT nprobe=8', marker='o')
plt.plot(recalls_FLAT_nprobe64[1, 0], qps_FLAT_nprobe64[1], label='FLAT nprobe=64', marker='o')
plt.plot(recalls_IVF_FLAT_nprobe8[1, 0], qps_IVF_FLAT_nprobe8[1], label='IVF_FLAT nprobe=8', marker='o')
plt.plot(recalls_IVF_FLAT_nprobe64[1, 0], qps_IVF_FLAT_nprobe64[1], label='IVF_FLAT nprobe=64', marker='o')
plt.plot(recalls_IVF_SQ8_nprobe8[1, 0], qps_IVF_SQ8_nprobe8[1], label='IVF_SQ8 nprobe=8', marker='o')
plt.plot(recalls_IVF_SQ8_nprobe64[1, 0], qps_IVF_SQ8_nprobe64[1], label='IVF_SQ8 nprobe=64', marker='o')
plt.plot(recalls_IVF_PQ_nprobe8[1, 0], qps_IVF_PQ_nprobe8[1], label='IVF_PQ nprobe=8', marker='o')
plt.plot(recalls_IVF_PQ_nprobe64[1, 0], qps_IVF_PQ_nprobe64[1], label='IVF_PQ nprobe=64', marker='o')
plt.plot(recalls_HNSW_nprobe8[1, 0], qps_HNSW_nprobe8[1], label='HNSW nprobe=8', marker='o')
plt.plot(recalls_HNSW_nprobe64[1, 0], qps_HNSW_nprobe64[1], label='HNSW nprobe=64', marker='o')
plt.plot(recalls_ANNOY_nprobe8[1, 0], qps_ANNOY_nprobe8[1], label='ANNOY nprobe=8', marker='o')
plt.plot(recalls_ANNOY_nprobe64[1, 0], qps_ANNOY_nprobe64[1], label='ANNOY nprobe=64', marker='o')

# recalls_FLAT_nprobe8[0, 0] = 0.9927
# write values at each point in format (qps, recall@1)
plt.text(recalls_FLAT_nprobe8[1, 0], qps_FLAT_nprobe8[1], '({:.4f}, {:.4f})'.format(recalls_FLAT_nprobe8[1, 0], qps_FLAT_nprobe8[1]))
plt.text(recalls_FLAT_nprobe64[1, 0], qps_FLAT_nprobe64[1], '({:.4f}, {:.4f})'.format(recalls_FLAT_nprobe64[1, 0], qps_FLAT_nprobe64[1]))
plt.text(recalls_IVF_FLAT_nprobe8[1, 0], qps_IVF_FLAT_nprobe8[1], '({:.4f}, {:.4f})'.format(recalls_IVF_FLAT_nprobe8[1, 0], qps_IVF_FLAT_nprobe8[1]))
plt.text(recalls_IVF_FLAT_nprobe64[1, 0], qps_IVF_FLAT_nprobe64[1], '({:.4f}, {:.4f})'.format(recalls_IVF_FLAT_nprobe64[1, 0], qps_IVF_FLAT_nprobe64[1]))
plt.text(recalls_IVF_SQ8_nprobe8[1, 0], qps_IVF_SQ8_nprobe8[1], '({:.4f}, {:.4f})'.format(recalls_IVF_SQ8_nprobe8[1, 0], qps_IVF_SQ8_nprobe8[1]))
plt.text(recalls_IVF_SQ8_nprobe64[1, 0], qps_IVF_SQ8_nprobe64[1], '({:.4f}, {:.4f})'.format(recalls_IVF_SQ8_nprobe64[1, 0], qps_IVF_SQ8_nprobe64[1]))
plt.text(recalls_IVF_PQ_nprobe8[1, 0], qps_IVF_PQ_nprobe8[1], '({:.4f}, {:.4f})'.format(recalls_IVF_PQ_nprobe8[1, 0], qps_IVF_PQ_nprobe8[1]))
plt.text(recalls_IVF_PQ_nprobe64[1, 0], qps_IVF_PQ_nprobe64[1], '({:.4f}, {:.4f})'.format(recalls_IVF_PQ_nprobe64[1, 0], qps_IVF_PQ_nprobe64[1]))
plt.text(recalls_HNSW_nprobe8[1, 0], qps_HNSW_nprobe8[1], '({:.4f}, {:.4f})'.format(recalls_HNSW_nprobe8[1, 0], qps_HNSW_nprobe8[1]))
plt.text(recalls_HNSW_nprobe64[1, 0], qps_HNSW_nprobe64[1], '({:.4f}, {:.4f})'.format(recalls_HNSW_nprobe64[1, 0], qps_HNSW_nprobe64[1]))
plt.text(recalls_ANNOY_nprobe8[1, 0], qps_ANNOY_nprobe8[1], '({:.4f}, {:.4f})'.format(recalls_ANNOY_nprobe8[1, 0], qps_ANNOY_nprobe8[1]))
plt.text(recalls_ANNOY_nprobe64[1, 0], qps_ANNOY_nprobe64[1], '({:.4f}, {:.4f})'.format(recalls_ANNOY_nprobe64[1, 0], qps_ANNOY_nprobe64[1]))

plt.xlabel('recall@1')
plt.ylabel('qps')
plt.title('qps vs recall@1 for nq=10')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.savefig('qps_vs_recall_nq10.png', bbox_inches='tight')
plt.close()

# nq = 100
plt.figure(figsize=(18, 10))
# plt.plot(recalls_FLAT_nprobe8[2, 0], qps_FLAT_nprobe8[2], label='FLAT nprobe=8', marker='o')
# plt.plot(recalls_FLAT_nprobe64[2, 0], qps_FLAT_nprobe64[2], label='FLAT nprobe=64', marker='o')
plt.plot(recalls_IVF_FLAT_nprobe8[2, 0], qps_IVF_FLAT_nprobe8[2], label='IVF_FLAT nprobe=8', marker='o')
plt.plot(recalls_IVF_FLAT_nprobe64[2, 0], qps_IVF_FLAT_nprobe64[2], label='IVF_FLAT nprobe=64', marker='o')
plt.plot(recalls_IVF_SQ8_nprobe8[2, 0], qps_IVF_SQ8_nprobe8[2], label='IVF_SQ8 nprobe=8', marker='o')
plt.plot(recalls_IVF_SQ8_nprobe64[2, 0], qps_IVF_SQ8_nprobe64[2], label='IVF_SQ8 nprobe=64', marker='o')
plt.plot(recalls_IVF_PQ_nprobe8[2, 0], qps_IVF_PQ_nprobe8[2], label='IVF_PQ nprobe=8', marker='o')
plt.plot(recalls_IVF_PQ_nprobe64[2, 0], qps_IVF_PQ_nprobe64[2], label='IVF_PQ nprobe=64', marker='o')
plt.plot(recalls_HNSW_nprobe8[2, 0], qps_HNSW_nprobe8[2], label='HNSW nprobe=8', marker='o')
plt.plot(recalls_HNSW_nprobe64[2, 0], qps_HNSW_nprobe64[2], label='HNSW nprobe=64', marker='o')
plt.plot(recalls_ANNOY_nprobe8[2, 0], qps_ANNOY_nprobe8[2], label='ANNOY nprobe=8', marker='o')
plt.plot(recalls_ANNOY_nprobe64[2, 0], qps_ANNOY_nprobe64[2], label='ANNOY nprobe=64', marker='o')

# recalls_FLAT_nprobe8[0, 0] = 0.9927
# write values at each point in format (qps, recall@1)
plt.text(recalls_IVF_FLAT_nprobe8[2, 0], qps_IVF_FLAT_nprobe8[2], '({:.4f}, {:.4f})'.format(recalls_IVF_FLAT_nprobe8[2, 0], qps_IVF_FLAT_nprobe8[2]))
plt.text(recalls_IVF_FLAT_nprobe64[2, 0], qps_IVF_FLAT_nprobe64[2], '({:.4f}, {:.4f})'.format(recalls_IVF_FLAT_nprobe64[2, 0], qps_IVF_FLAT_nprobe64[2]))
plt.text(recalls_IVF_SQ8_nprobe8[2, 0], qps_IVF_SQ8_nprobe8[2], '({:.4f}, {:.4f})'.format(recalls_IVF_SQ8_nprobe8[2, 0], qps_IVF_SQ8_nprobe8[2]))
plt.text(recalls_IVF_SQ8_nprobe64[2, 0], qps_IVF_SQ8_nprobe64[2], '({:.4f}, {:.4f})'.format(recalls_IVF_SQ8_nprobe64[2, 0], qps_IVF_SQ8_nprobe64[2]))
plt.text(recalls_IVF_PQ_nprobe8[2, 0], qps_IVF_PQ_nprobe8[2], '({:.4f}, {:.4f})'.format(recalls_IVF_PQ_nprobe8[2, 0], qps_IVF_PQ_nprobe8[2]))
plt.text(recalls_IVF_PQ_nprobe64[2, 0], qps_IVF_PQ_nprobe64[2], '({:.4f}, {:.4f})'.format(recalls_IVF_PQ_nprobe64[2, 0], qps_IVF_PQ_nprobe64[2]))
plt.text(recalls_HNSW_nprobe8[2, 0], qps_HNSW_nprobe8[2], '({:.4f}, {:.4f})'.format(recalls_HNSW_nprobe8[2, 0], qps_HNSW_nprobe8[2]))
plt.text(recalls_HNSW_nprobe64[2, 0], qps_HNSW_nprobe64[2], '({:.4f}, {:.4f})'.format(recalls_HNSW_nprobe64[2, 0], qps_HNSW_nprobe64[2]))
plt.text(recalls_ANNOY_nprobe8[2, 0], qps_ANNOY_nprobe8[2], '({:.4f}, {:.4f})'.format(recalls_ANNOY_nprobe8[2, 0], qps_ANNOY_nprobe8[2]))
plt.text(recalls_ANNOY_nprobe64[2, 0], qps_ANNOY_nprobe64[2], '({:.4f}, {:.4f})'.format(recalls_ANNOY_nprobe64[2, 0], qps_ANNOY_nprobe64[2]))

plt.xlabel('recall@1')
plt.ylabel('qps')
plt.title('qps vs recall@1 for nq=100')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.savefig('qps_vs_recall_nq100.png', bbox_inches='tight')
plt.close()
