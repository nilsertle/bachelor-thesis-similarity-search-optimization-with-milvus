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
plt.title('TPQ vs nq for IVF_FLAT')
plt.legend()
plt.savefig('Graphs/saves/IVF_FLAT.png')
plt.clf()

''' IVF_SQ8 '''
x_nq = [1, 10, 100, 300, 500, 700, 900]
y1_tpq = [0.08370630741119385, 0.182344753742218, 0.5950490546226501, 0.2858625864982605, 1.5372262716293335, 2.3194054555892945, 1.1452929162979126]
y2_tpq = [0.02129079580307007, 0.07418950080871582, 0.720581521987915, 0.9841989326477051, 6.178641271591187, 0.5834645485877991, 1.7034779477119446]
y3_tpq = [0.1531834149360657, 0.6566210389137268, 1.8879489850997926, 4.57451711177826, 7.299325003623962, 6.306681277751923, 6.706688194274903]
y4_tpq = [0.053653230667114256, 0.17572896003723146, 5.075315713882446, 11.64424933195114, 11.600434536933898, 16.18816546201706, 20.942028772830962]

plt.plot(x_nq, y1_tpq, label='256 / 8', marker='o')
plt.plot(x_nq, y2_tpq, label='1024 / 32', marker='o')
plt.plot(x_nq, y3_tpq, label='4096 / 128', marker='o')
plt.plot(x_nq, y4_tpq, label='16384 / 512', marker='o')

plt.xlabel('nq')
plt.ylabel('TPQ / s')
plt.title('TPQ vs nq for IVF_SQ8')
plt.legend()
plt.savefig('Graphs/saves/IVF_SQ8.png')
plt.clf()

''' FLAT '''
x_nq = [1, 10, 100, 300, 500, 700, 900]
y1_tpq = [0.04909815073013306, 0.15259602308273315, 4.08134397983551, 13.904570820331573, 22.75145353794098, 33.63976550102234, 42.39939521074295]
y2_tpq = [0.03127134084701538, 0.16023435592651367, 3.0300535345077515, 13.827951004505158, 23.041417076587678, 34.437500557899476, 49.41446811437607]
y3_tpq = [0.06604260444641114, 0.16707286834716797, 2.4467604327201844, 15.71842530488968, 26.252528293132784, 43.64203419208527, 51.82040724992752]
y4_tpq = [0.0657002067565918, 0.19498781681060792, 1.7316991806030273, 5.026195983886719, 11.271984322071075, 20.793392570018767, 51.75009387731552]

plt.plot(x_nq, y1_tpq, label='256 / 8', marker='o')
plt.plot(x_nq, y2_tpq, label='1024 / 32', marker='o')
plt.plot(x_nq, y3_tpq, label='4096 / 128', marker='o')
plt.plot(x_nq, y4_tpq, label='16384 / 512', marker='o')

plt.xlabel('nq')
plt.ylabel('TPQ / s')
plt.title('TPQ vs nq for FLAT')
plt.legend()
plt.savefig('Graphs/saves/FLAT.png')
plt.clf()