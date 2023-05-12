import numpy as np
from tensorflow import keras
import ncps as kncp
from ncps.wirings import AutoNCP
from ncps.tf import LTC
import matplotlib.pyplot as plt
import seaborn as sns

# Download the dataset (already implemented in keras-ncp)
(
    (x_train, y_train),
    (x_valid, y_valid),
) = kncp.datasets.icra2020_lidar_collision_avoidance.load_data()
print("x_train", str(x_train.shape))
print("y_train", str(y_train.shape))


# Plot the data
def plot_lidar(lidar, ax):
    # Helper function for plotting polar-based lidar data
    angles = np.linspace(-2.35, 2.35, len(lidar))
    x = lidar * np.cos(angles)
    y = lidar * np.sin(angles)
    ax.plot(y, x)
    ax.scatter([0], [0], marker="^", color="black")
    ax.set_xlim((-6, 6))
    ax.set_ylim((-2, 6))


sns.set()
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
plot_lidar(x_train[0, 0, :, 0], ax1)
plot_lidar(x_train[0, 12, :, 0], ax2)
plot_lidar(x_train[9, 0, :, 0], ax3)
ax1.set_title("Label: {:0.2f}".format(y_train[0, 0, 0]))
ax2.set_title("Label: {:0.2f}".format(y_train[0, 12, 0]))
ax3.set_title("Label: {:0.2f}".format(y_train[9, 0, 0]))
fig.suptitle("LIDAR collision avoidance training examples")


# Build the network

N = x_train.shape[2]
channels = x_train.shape[3]
neuron_numbers = [6, 7, 8, 10, 11, 13, 15, 17, 19, 21]

for neuron_number in neuron_numbers:
    wiring = AutoNCP(neuron_number,1)

    # We need to use the TimeDistributed layer to independently apply the
    # Conv1D/MaxPool1D/Dense over each time-step of the input time-series.
    model = keras.models.Sequential(
        [
            keras.layers.InputLayer(input_shape=(None, N, channels)),
            keras.layers.TimeDistributed(
                keras.layers.Conv1D(18, 5, strides=3, activation="relu")
            ),
            keras.layers.TimeDistributed(
                keras.layers.Conv1D(20, 5, strides=2, activation="relu")
            ),
            keras.layers.TimeDistributed(keras.layers.MaxPool1D()),
            keras.layers.TimeDistributed(
                keras.layers.Conv1D(22, 5, activation="relu")
            ),
            keras.layers.TimeDistributed(keras.layers.MaxPool1D()),
            keras.layers.TimeDistributed(
                keras.layers.Conv1D(24, 5, activation="relu")
            ),
            keras.layers.TimeDistributed(keras.layers.Flatten()),
            keras.layers.TimeDistributed(keras.layers.Dense(32, activation="relu")),
            LTC(wiring, return_sequences=True),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(0.01), loss="mean_squared_error",
    )

    model.summary(line_length=100)

    # Plot the network architecture

    sns.set_style("white")
    plt.figure(figsize=(12, 12))
    legend_handles = wiring.draw_graph(layout='spiral',neuron_colors={"command": "tab:cyan"})
    plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(1, 1))
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig("architecture" + str(neuron_number) + ".png")

# Evaluate the model before training
# print("Validation set MSE before training")
# model.evaluate(x_valid, y_valid)

# # Train the model
# model.fit(
#     x=x_train, y=y_train, batch_size=32, epochs=20, validation_data=(x_valid, y_valid)
# )

# # Evaluate the model again after the training
# print("Validation set MSE after training")
# model.evaluate(x_valid, y_valid)
