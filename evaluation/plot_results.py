import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


data_removed = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
# data = {
#     "Random": [0.340, 0.349, 0.340, 0.342, 0.346, 0.346, 0.356, 0.351, 0.351, 0.350],
#     "DataOob": [0.340, 0.344, 0.341, 0.343, 0.345, 0.345, 0.356, 0.359, 0.357, 0.357],
#     "DataShapley": [0.340, 0.345, 0.345, 0.350, 0.351, 0.352, 0.346, 0.352, 0.354, 0.356],
#     "KNNShapley": [0.340, 0.338, 0.346, 0.347, 0.348, 0.350, 0.348, 0.354, 0.357, 0.354],
#     "TimeInf": [0.340, 0.343, 0.341, 0.343, 0.347, 0.355, 0.350, 0.350, 0.353, 0.353],
#     "TSRating": [0.340, 0.342, 0.347, 0.349, 0.354, 0.358, 0.361, 0.363, 0.371, 0.373]
# }

# data = {
#     "Random": [1.729, 1.735, 1.773, 1.782, 1.826, 1.850, 1.918, 1.900, 1.985, 2.060],
#     "DataOob": [1.729, 1.769, 1.791, 1.816, 1.811, 1.860, 1.908, 1.958, 1.960, 2.108],
#     "DataShapley": [1.729, 1.757, 1.738, 1.780, 1.870, 1.903, 1.952, 2.009, 2.025, 2.072],
#     "KNNShapley": [1.729, 1.755, 1.792, 1.795, 1.849, 1.897, 1.900, 1.958, 2.046, 2.117],
#     "TimeInf": [1.729, 1.771, 1.829, 1.814, 1.869, 1.870, 1.936, 1.892, 1.998, 2.023],
#     "TSRating": [1.729, 1.739, 1.786, 1.798, 1.846, 1.959, 2.000, 2.059, 2.076, 2.125]
# }

data = {
    "Random": [0.735, 0.711, 0.691, 0.684, 0.656, 0.629, 0.606, 0.574, 0.573, 0.570],
    "DataOob": [0.735, 0.737, 0.690, 0.702, 0.659, 0.627, 0.636, 0.612, 0.609, 0.594],
    "DataShapley": [0.735, 0.674, 0.657, 0.640, 0.638, 0.633, 0.599, 0.572, 0.562, 0.559],
    "KNNShapley": [0.735, 0.694, 0.666, 0.652, 0.637, 0.611, 0.598, 0.586, 0.584, 0.536],
    "TimeInf": [0.735, 0.730, 0.724, 0.722, 0.700, 0.666, 0.650, 0.620, 0.579, 0.539],
    "TSRating": [0.735, 0.712, 0.682, 0.639, 0.637, 0.607, 0.581, 0.562, 0.547, 0.500]
}


plt.figure(figsize=(10, 6))
for score_key, mae_values in data.items():
    plt.plot(data_removed, mae_values, marker='o', label=score_key)


plt.xlabel('% Data Removed', fontsize=25)
plt.ylabel('Accuracy', fontsize=25)
plt.title('CBF with Nonstationary_Transformer', fontsize=28)
plt.legend(title="Method", title_fontsize=16, fontsize=18)
plt.grid(alpha=0.5)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(data_removed, labels=[f"{int(p * 100)}%" for p in data_removed])

# save plots to results_plots folder
output_folder = 'results_plots'
os.makedirs(output_folder, exist_ok=True)
file_name = 'CBF with Nonstationary_Transformer'
plt.tight_layout()
plt.savefig(os.path.join(output_folder, file_name))

plt.show()
