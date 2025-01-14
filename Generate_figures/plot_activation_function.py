import numpy as np
import torch

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
sns.set_context("talk")
sns.set(rc={"xtick.bottom": True, "ytick.left": True})
colors = sns.color_palette("mako").as_hex()
my_cmap = sns.color_palette("viridis", as_cmap=True)


xx = np.linspace(-3, 3, 200)

plt.figure(figsize=(4, 3))
plt.plot(xx, torch.nn.functional.relu(torch.tensor(xx)), label=r"$RELU(x)$")
plt.legend()
plt.tight_layout()
plt.savefig("relu.pdf")
plt.show()

plt.figure(figsize=(4, 3))
plt.plot(xx, torch.nn.functional.gelu(torch.tensor(xx)), label=r"$GELU(x)$")
plt.legend()
plt.tight_layout()
plt.savefig("gelu.pdf")
plt.show()


# plt.subplot(3, 1, 2)
# plt.plot(xx, torch.nn.functional.gelu(torch.tensor(xx)), label=r"$GELU(x)$")
# plt.legend()
# plt.subplot(3, 1, 3)
# plt.plot(xx, torch.sin(torch.tensor(xx)), label=r"$\sin(x)$")
# plt.legend()
# plt.tight_layout()
# plt.savefig("activation_functions.pdf")
# plt.show()
