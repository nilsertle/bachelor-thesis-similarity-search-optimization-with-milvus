from matplotlib import pyplot as plt
import numpy as np

''' IVF_FLAT '''
x_nq = [1, 10, 100, 300, 500, 700, 900]
y1_tpq = [0.027623939514160156, 0.011418588161468506, 0.06675423860549927, 0.1487838315963745, 0.23962212324142457, 0.3441977143287659, 0.4365926456451416]
y2_tpq = [0.02418938398361206, 0.03110030174255371, 0.1548808431625366, 0.321646032333374, 0.4449701642990112, 0.4569703221321106, 0.5903023958206177]
y3_tpq = [0.031903331279754636, 0.07165949821472167, 0.5435235214233398, 0.9986433601379394, 1.092319793701172, 1.3385835146903993, 1.6829349780082703]
y4_tpq = [0.044682068824768065, 0.19153385162353515, 1.3663459920883179, 2.893317685127258, 8.891230101585387, 14.6459153008461, 19.156358132362367]

plt.plot(x_nq, y1_tpq, label='256 / 8', marker='o')
plt.plot(x_nq, y2_tpq, label='1024 / 32', marker='o')
plt.plot(x_nq, y3_tpq, label='4096 / 128', marker='o')
plt.plot(x_nq, y4_tpq, label='16384 / 512', marker='o')

plt.xlabel('nq')
plt.ylabel('TPQ / s')
plt.title('TPQ vs nq for IVF_FLAT / s')
plt.legend()
plt.show()

''' IVF_SQ8 '''
x_nq = [1, 10, 100, 300, 500, 700, 900]
y1_tpq = []
y2_tpq = []
y3_tpq = []
y4_tpq = []

plt.plot(x_nq, y1_tpq, label='256 / 8', marker='o')
plt.plot(x_nq, y2_tpq, label='1024 / 32', marker='o')
plt.plot(x_nq, y3_tpq, label='4096 / 128', marker='o')
plt.plot(x_nq, y4_tpq, label='16384 / 512', marker='o')

plt.xlabel('nq')
plt.ylabel('TPQ / s')
plt.title('TPQ vs nq for IVF_SQ8 / s')
plt.legend()
plt.show()