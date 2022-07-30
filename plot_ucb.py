import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

num_agents = [2, 4, 6, 8, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

#finite horizon 20 10 20 10
# regret_20_10_20_10 = [214.2771759033203, 216.0682830810547, 212.64797973632812, 191.7488250732422, 188.2528533935547,
#                       187.37191772460938, 187.97396850585938, 182.68576049804688, 180.5601806640625, 184.35873413085938,
#                       179.6865692138672, 177.02833557128906, 178.1661834716797, 180.20999145507812]
#
# psrl_20_10_20_10 = np.array(pd.read_csv("/Users/perrydong/Desktop/z/concurrent_psrl/results/finite/20_10.csv", index_col=None, header=None))
#
# plt.plot(num_agents, regret_20_10_20_10, label="Cooperative UCB", color="green")
# plt.plot(num_agents, psrl_20_10_20_10[:, 1]*200, label="Cooperative PSRL", color="blue")
# plt.title("Cooperative PSRL vs Cooperative UCB Finite Horizon \n S=20 A=10 K=20 H=10", linespacing=1.5)

#finite horizon 20 10 20 10
# regret_5_5_20_10 = [182.37313842773438, 167.13778686523438, 158.83456420898438, 155.68777465820312, 162.72195434570312,
#                     147.68264770507812, 153.50051879882812, 160.73056030273438, 157.99830627441406, 160.243896484375,
#                     159.1931610107422, 154.7383575439453, 160.92587280273438, 154.88565063476562, 157.15237426757812]
# psrl_5_5_20_10 = np.array(pd.read_csv("/Users/perrydong/Desktop/z/concurrent_psrl/results/finite/5_5_20_10.csv", index_col=None, header=None))
# plt.plot([1] + num_agents, regret_5_5_20_10, label="Cooperative UCB", color="green")
# plt.plot(num_agents, psrl_5_5_20_10[:, 1]*200, label="Cooperative PSRL", color="blue")
# plt.title("Cooperative PSRL vs Cooperative UCB Finite Horizon \n S=5 A=5 K=5 H=5", linespacing=1.5)

#finite horizon 5 5 30 75
# regret_5_5_30_75 = [2075.992919921875, 1906.6529541015625, 1753.1494140625, 1734.208984375, 1796.201171875,
#                     1974.4447021484375, 1952.086181640625, 1774.510498046875, 1757.6090087890625, 1767.352783203125,
#                     1780.826416015625, 1767.5738525390625, 1763.217041015625, 1764.2503662109375, 1776.156982421875]
# psrl_5_5_30_75 = np.array(pd.read_csv("/Users/perrydong/Desktop/z/concurrent_psrl/results/finite/S5-A5-K30-H75.csv", index_col=None, header=None))
# plt.plot([1] + num_agents, regret_5_5_30_75, label="Cooperative UCB", color="green")
# plt.plot(num_agents, psrl_5_5_30_75[:, 1]*2250, label="Cooperative PSRL", color="blue")
# plt.title("Cooperative PSRL vs Cooperative UCB Finite Horizon \n S=5 A=5 K=30 H=75", linespacing=1.5)

#finite horizon 20 10 30 75
# regret_20_10_30_75 = [2389.33837890625, 2183.75439453125, 2280.30517578125, 2308.96533203125, 2230.316162109375,
#                       2220.061279296875, 2200.150390625, 2163.20703125, 2144.267578125, 2157.44091796875,
#                       2178.410400390625, 2180.25390625, 2167.179931640625, 2167.173828125, 2164.35302734375]
# psrl_20_10_30_75= [0.2784, 0.1895, 0.1454, 0.1123, 0.0971, 0.0649, 0.0502, 0.0471, 0.0411, 0.0438, 0.0403, 0.0440, 0.0385, 0.0381]
# plt.plot([1] + num_agents, regret_20_10_30_75, label="Cooperative UCB", color="green")
# plt.plot(num_agents, np.array(psrl_20_10_30_75)*2250, label="Cooperative PSRL", color="blue")
# plt.title("Cooperative PSRL vs Cooperative UCB Finite Horizon \n S=20 A=10 K=30 H=75", linespacing=1.5)

#infinite horizon 1000
# regret_1000 = [770.2685546875, 698.5173950195312, 750.8216552734375, 760.6071166992188, 722.1397094726562,
#                680.8868408203125, 770.4791870117188, 664.9152221679688, 689.7868041992188]
# psrl_1000 = np.array(pd.read_csv("/Users/perrydong/Desktop/z/concurrent_psrl/results/infinite/infinite2_seed_100_S_10_A_5_T_1000_agents_50.csv", index_col=None, header=None))
# plt.plot([1] + num_agents[:len(regret_1000)-1], np.array(regret_1000), label="Cooperative UCB", color="green")
# plt.plot(psrl_1000[:-1, 0], psrl_1000[:-1, 1], label="Cooperative PSRL", color='navy')
# plt.title("Cooperative PSRL vs Cooperative UCB Infinite Horizon \n S=10 A=5 T=1000", linespacing=1.5)

#infinite horizon 2000
regret_2000 = [1543.5628662109375, 1494.1256103515625, 1549.0980224609375, 1537.93798828125, 1511.7398681640625,
               1455.4329833984375, 1433.03271484375, 1535.898193359375, 1428.227294921875]
psrl_2000 = np.array(pd.read_csv("/Users/perrydong/Desktop/z/concurrent_psrl/results/infinite/infinite2_seed_100_S_10_A_5_T_2000_agents_50.csv", index_col=None, header=None))
plt.plot([1] + num_agents[:len(regret_2000)-1], np.array(regret_2000), label="Cooperative UCB", color="green")
plt.plot(psrl_2000[:-1, 0], psrl_2000[:-1, 1], label="Cooperative PSRL", color='navy')
plt.title("Cooperative PSRL vs Cooperative UCB Infinite Horizon \n S=10 A=5 T=2000", linespacing=1.5)

plt.xlabel("Number of Agents")
plt.ylabel("Regret")
plt.legend()
plt.show()