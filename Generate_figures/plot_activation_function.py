import numpy as np
import matplotlib.pyplot as plt
import random
import os
import seaborn as sns

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

seed = 2023
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.experimental.numpy.random.seed(seed)
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
# Set a fixed value for the hash seed
os.environ["PYTHONHASHSEED"] = str(seed)

sns.set_theme()
sns.set_context("paper")
sns.set(rc={"xtick.bottom": True, "ytick.left": True})
colors = sns.color_palette("mako").as_hex()
my_cmap = sns.color_palette("viridis", as_cmap=True)


x = np.linspace(-4.0, 4.0, 1000)
x = tf.constant(x, dtype=tf.float32)
y2 = tf.keras.activations.gelu(x).numpy()

plt.figure(figsize=(6, 3))
plt.plot(x, y2, label="$gelu(x)$")
plt.legend()
plt.tight_layout()
plt.savefig("./activation_functions.png")
plt.show()
