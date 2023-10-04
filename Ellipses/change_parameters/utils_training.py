import numpy as np
import matplotlib.pyplot as plt
import random
import os

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
# import dolfin as df
import time
from utils import *
import seaborn as sns

sns.set_theme()
sns.set_context("paper")
sns.set(rc={"xtick.bottom": True, "ytick.left": True})
colors = sns.color_palette("mako").as_hex()
my_cmap = sns.color_palette("viridis", as_cmap=True)


class DataLoader:
    """
    DataLoader class for loading and processing data.
    """

    def __init__(self, small_data=False, dtype=tf.float32):
        """
        Initialize the DataLoader object.

        Parameters:
            small_data (bool, optional): Whether to use a smaller dataset. Defaults to False.
            dtype (tf.DType, optional): Data type to use for loading data. Defaults to tf.float32.
        """
        F = tf.constant(np.load(f"../data/F.npy"), dtype=dtype)
        self.max_norm_F = tf.constant(
            np.max(np.sqrt(np.mean(F**2, axis=(1, 2)))), dtype=dtype
        )
        print("max_norm_F :", self.max_norm_F.numpy())

        F = tf.constant(
            np.load(f"../data/F.npy") / self.max_norm_F,
            dtype=dtype,
        )

        Phi = tf.constant(np.load(f"../data/Phi.npy"), dtype=dtype)
        G = tf.constant(
            np.load(f"../data/G.npy"),
            dtype=dtype,
        )

        Y = tf.constant(np.load(f"../data/W.npy"), dtype=dtype)[:, :, :, None]

        if small_data:
            data_size = 100
        else:
            data_size = F.shape[0]

        if small_data:
            F = F[:data_size]
            Phi = Phi[:data_size]
            G = G[:data_size]
            Y = Y[:data_size]

        nb_vert = F.shape[1]
        domain, boundary = create_domain_and_boundary(nb_vert, Phi)
        domain, boundary = tf.constant(domain, dtype=dtype), tf.constant(
            boundary, dtype=dtype
        )

        X = tf.stack([F, Phi, G, domain], axis=3)
        self.input_shape = (None, X.shape[1], X.shape[2], X.shape[3])

        nb_val = data_size // 8
        nb_train = data_size - nb_val
        print("data_size,nb_val,nb_train:", data_size, nb_val, nb_train)
        print("data shape:", self.input_shape)

        def separe(A):
            return A[:nb_train], A[nb_train:]

        self.X_train, self.X_val = separe(X)
        self.Y_train, self.Y_val = separe(Y)
        self.nb_vert = self.X_train.shape[1]


def reflexive_padding_2d(W: tf.Tensor, pad_1: int, pad_2: int):
    """
    Apply reflexive padding to a 2D tensor.

    Parameters:
        W (tf.Tensor): Input tensor of shape (batch_size, nx, ny, channels).
        pad_1 (int): Padding size along the first dimension.
        pad_2 (int): Padding size along the second dimension.

    Returns:
        tf.Tensor: The padded tensor.
    """
    return tf.pad(
        W, [[0, 0], [pad_1, pad_1], [pad_2, pad_2], [0, 0]], mode="REFLECT"
    )


class SpectralConv2d(tf.keras.layers.Layer):
    """
    SpectralConv2d layer for a FNO.
    """

    def __init__(self, in_channels, out_channels, modes1, modes2):
        """
        Initialize the SpectralConv2d layer.

        Parameters:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            modes1 (int): Number of Fourier modes to multiply along the first dimension.
            modes2 (int): Number of Fourier modes to multiply along the second dimension.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N_/2) + 1
        self.modes2 = modes2

        self.weights1 = self.get_complex_weights_2d(in_channels, out_channels)
        self.weights2 = self.get_complex_weights_2d(in_channels, out_channels)

    def get_complex_weights_2d(self, in_channels, out_channels):
        scale = 1 / (in_channels * out_channels)
        real = tf.random.uniform(
            [in_channels, out_channels, self.modes1, self.modes2]
        )
        img = tf.random.uniform(
            [in_channels, out_channels, self.modes1, self.modes2]
        )
        return tf.Variable(tf.complex(real, img) * scale)

    def channel_mix(self, inputs, weights):
        # (batch, in_channel, nx,ny ), (in_channel, out_channel, nx) -> (batch, out_channel, ny)
        return tf.einsum("bixy,ioxy->boxy", inputs, weights)

    def call(self, A):
        A = tf.transpose(A, [0, 3, 1, 2])

        # Compute Fourier coeffcients
        A_ft = tf.signal.rfft2d(A)  # shape=(...,A.shape[-2],A.shape[-1]//2+1)

        # Multiply relevant Fourier modes
        out_ft_corner_SO = self.channel_mix(
            A_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        out_ft_corner_NO = self.channel_mix(
            A_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )

        out_ft_corner_SO = tf.pad(
            out_ft_corner_SO,
            [
                [0, 0],
                [0, 0],
                [0, A.shape[2] - self.modes1],
                [0, A_ft.shape[3] - self.modes2],
            ],
        )
        out_ft_corner_NO = tf.pad(
            out_ft_corner_NO,
            [
                [0, 0],
                [0, 0],
                [A.shape[2] - self.modes1, 0],
                [0, A_ft.shape[3] - self.modes2],
            ],
        )
        out_ft = out_ft_corner_SO + out_ft_corner_NO
        # Return to physical space
        A = tf.signal.irfft2d(out_ft, fft_length=[A.shape[2], A.shape[3]])

        # channel axis goes back at the end
        A = tf.transpose(A, [0, 2, 3, 1])

        return A


class FNO2d(tf.keras.Model):
    """Implementation of the FNO model."""

    def __init__(self, modes: int, width: int, pad_prop=0.05, nb_layers=4):
        """
        Initialize the FNO2d model.

        Parameters:
            modes (int): Number of Fourier modes to use in the spectral convolution.
            width (int): Width of the FNO2d model.
            pad_prop (float, optional): Proportion of padding to apply. Defaults to 0.05.
        """
        super().__init__()

        self.modes1 = modes
        self.modes2 = modes
        self.width = width

        self.pad_prop = pad_prop

        print(
            f"FNO2d model created with hyperparameters : modes:{modes}, width:{width}, pad_prop:{self.pad_prop} "
        )

        self.fc0 = tf.keras.layers.Dense(
            self.width
        )  # input channel is dim_a+1: (a(A), A)

        nb_layer = nb_layers
        self.convs = [
            SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
            for _ in range(nb_layer)
        ]
        self.addis = [
            tf.keras.layers.Conv2D(self.width, 1, padding="SAME")
            for _ in range(nb_layer)
        ]

        self.fc1 = tf.keras.layers.Dense(128)
        self.fc2 = tf.keras.layers.Dense(1)

    @tf.function
    def call(self, A):
        pad_1 = int(A.shape[1] * self.pad_prop)
        pad_2 = int(A.shape[2] * self.pad_prop)

        A = reflexive_padding_2d(A, pad_1, pad_2)

        A = self.fc0(A)

        for i, (layer, addi) in enumerate(zip(self.convs, self.addis)):
            x1 = layer.call(A)
            x2 = addi(A)
            A = x1 + x2
            if i != len(self.convs) - 1:
                A = tf.nn.gelu(A)

        A = A[:, pad_1:-pad_1, pad_2:-pad_2, :]
        A = self.fc1(A)
        A = tf.nn.gelu(A)
        A = self.fc2(A)

        return A


class Agent:
    """
    Agent class for training and evaluating the FNO2d model.
    """

    def __init__(
        self,
        data: DataLoader,
        small_model=False,
        width=20,
        modes=20,
        nb_layers=4,
        pad_prop=0.05,
        index_test_case=1,
        loss_level=2,
    ):
        self.nb_vert = data.nb_vert
        self.data = data
        self.index_test_case = index_test_case

        self.model = FNO2d(modes, width, pad_prop, nb_layers)
        self.learning_rate = 1e-3
        self.optimizer_misfit = tf.keras.optimizers.Adam(self.learning_rate)
        x = tf.linspace(0.0, 1.0, self.nb_vert)
        xx, yy = tf.meshgrid(x, x)
        self.x = tf.Variable(tf.reshape(xx, [-1]))
        self.y = tf.Variable(tf.reshape(yy, [-1]))

        self.loss_level = loss_level

        self.memo_misfit_0_train = []
        self.memo_misfit_0_val = []
        self.memo_misfit_1_train = []
        self.memo_misfit_1_val = []
        self.memo_misfit_2_train = []
        self.memo_misfit_2_val = []
        self.memo_loss_train = []
        self.memo_loss_val = []
        self.memo_val_steps = []
        self.memo_optimize_misfit = []

        self.derivator_level1 = Derivator_fd(
            axes=[1, 2],
            interval_lenghts=[1, 1],
            derivative_symbols=[(0,), (1,)],
        )
        self.derivator_level2 = Derivator_fd(
            axes=[1, 2],
            interval_lenghts=[1, 1],
            derivative_symbols=[(0,), (1,), (0, 0), (1, 1), (0, 1)],
        )  # {"x":(0,),"y":(1,),"xx":(0,0),"yy":(1,1),"xy":(0,1)}

    def H_loss(self, Y_true, Y_pred, Phi, level, domain, G):
        (
            Y_true_x,
            Y_true_y,
            Y_true_xx,
            Y_true_yy,
            Y_true_xy,
        ) = self.derivator_level2(
            Y_true * Phi[:, :, :, None] + G[:, :, :, None]
        )
        (
            Y_pred_x,
            Y_pred_y,
            Y_pred_xx,
            Y_pred_yy,
            Y_pred_xy,
        ) = self.derivator_level2(
            Y_pred * Phi[:, :, :, None] + G[:, :, :, None]
        )

        error_0 = tf.reduce_mean(
            (
                ((Y_true * Phi[:, :, :, None]) - (Y_pred * Phi[:, :, :, None]))
                * domain[:, :, :, None]
            )
            ** 2,
            axis=[1, 2, 3],
        )

        error_1 = (
            tf.reduce_mean(
                ((Y_true_x - Y_pred_x) * domain[:, :, :, None]) ** 2,
                axis=[1, 2, 3],
            )
        ) + (
            tf.reduce_mean(
                ((Y_true_y - Y_pred_y) * domain[:, :, :, None]) ** 2,
                axis=[1, 2, 3],
            )
        )

        error_2 = (
            (
                tf.reduce_mean(
                    ((Y_true_xx - Y_pred_xx) * domain[:, :, :, None]) ** 2,
                    axis=[1, 2, 3],
                )
            )
            + (
                tf.reduce_mean(
                    ((Y_true_xy - Y_pred_xy) * domain[:, :, :, None]) ** 2,
                    axis=[1, 2, 3],
                )
            )
            + (
                tf.reduce_mean(
                    ((Y_true_yy - Y_pred_yy) * domain[:, :, :, None]) ** 2,
                    axis=[1, 2, 3],
                )
            )
        )

        magnitude_0 = tf.reduce_mean(
            (
                ((Y_true * Phi[:, :, :, None] + G[:, :, :, None]))
                * domain[:, :, :, None]
            )
            ** 2,
            axis=[1, 2, 3],
        )
        magnitude_1 = (
            tf.reduce_mean(
                ((Y_true_x) * domain[:, :, :, None]) ** 2,
                axis=[1, 2, 3],
            )
        ) + (
            tf.reduce_mean(
                ((Y_true_y) * domain[:, :, :, None]) ** 2,
                axis=[1, 2, 3],
            )
        )
        magnitude_2 = (
            (
                tf.reduce_mean(
                    ((Y_true_xx) * domain[:, :, :, None]) ** 2,
                    axis=[1, 2, 3],
                )
            )
            + (
                tf.reduce_mean(
                    ((Y_true_xy) * domain[:, :, :, None]) ** 2,
                    axis=[1, 2, 3],
                )
            )
            + (
                tf.reduce_mean(
                    ((Y_true_yy) * domain[:, :, :, None]) ** 2,
                    axis=[1, 2, 3],
                )
            )
        )
        if level == 0:
            error = error_0
            magnitude = magnitude_0
            loss_0 = tf.reduce_mean(tf.sqrt(error_0 / magnitude_0))
            loss_1, loss_2 = np.nan, np.nan
        elif level == 1:
            error = error_0 + error_1
            magnitude = magnitude_0 + magnitude_1
            loss_0 = tf.reduce_mean(tf.sqrt(error_0 / magnitude_0))
            loss_1 = tf.reduce_mean(tf.sqrt(error_1 / magnitude_1))
            loss_2 = np.nan
        else:
            error = error_0 + error_1 + error_2
            magnitude = magnitude_0 + magnitude_1 + magnitude_2
            loss_0 = tf.reduce_mean(tf.sqrt(error_0 / magnitude_0))
            loss_1 = tf.reduce_mean(tf.sqrt(error_1 / magnitude_1))
            loss_2 = tf.reduce_mean(tf.sqrt(error_2 / magnitude_2))

        loss = tf.reduce_mean(tf.sqrt(error / magnitude))
        return loss_0, loss_1, loss_2, loss

    @tf.function
    def validate(self):
        print("validate method tracing")
        Y_pred = self.model.call(self.data.X_val)
        misfit_0, misfit_1, misfit_2, loss = self.H_loss(
            self.data.Y_val,
            Y_pred,
            self.data.X_val[:, :, :, 1],
            level=self.loss_level,
            domain=self.data.X_val[:, :, :, -1],
            G=self.data.X_val[:, :, :, 2],
        )
        return (
            misfit_0,
            misfit_1,
            misfit_2,
            loss,
        )

    @tf.function
    def train_one_epoch_misfit_acc(self, X, Y, batch_size, nb_batch, level):
        """
        X = [F, Phi, G, domain]
        """
        print(f"tracing train_one_epoch_misfit_acc, level={level}")

        total_misfit_0 = 0
        total_misfit_1 = 0
        total_misfit_2 = 0
        total_loss = 0

        for i in range(nb_batch):
            sli = slice(i * batch_size, (i + 1) * batch_size)
            X_, Y_ = X[sli], Y[sli]

            with tf.GradientTape() as tape:
                y_pred = self.model.call(X_)
                misfit_0, misfit_1, misfit_2, loss = self.H_loss(
                    Y_,
                    y_pred,
                    X_[:, :, :, 1],
                    level,
                    X_[:, :, :, -1],
                    X_[:, :, :, 2],
                )

                tv = self.model.trainable_variables
                grad = tape.gradient(loss, tv)
                self.optimizer_misfit.apply_gradients(zip(grad, tv))

            total_misfit_0 += misfit_0
            total_misfit_1 += misfit_1
            total_misfit_2 += misfit_2
            total_loss += loss

        return (
            total_misfit_0 / nb_batch,
            total_misfit_1 / nb_batch,
            total_misfit_2 / nb_batch,
            total_loss / nb_batch,
        )

    def scheduler(self, epoch, lr, power, final):
        return lr * np.exp(-power * epoch)

    def train(self, epochs):
        try:
            original_learning_rate = self.learning_rate
            for i in range(epochs):
                self.learning_rate = self.scheduler(
                    i, original_learning_rate, 0.0012, 2000
                )
                self.optimizer_misfit.lr.assign(self.learning_rate)
                self.memo_optimize_misfit.append(i)
                self.train_one_epoch(self.loss_level, i)

                if i > 0 and (i + 1) % 10 == 0:
                    (
                        misfit_0,
                        misfit_1,
                        misfit_2,
                        loss,
                    ) = self.validate()
                    self.memo_misfit_0_val.append(misfit_0)
                    self.memo_misfit_1_val.append(misfit_1)
                    self.memo_misfit_2_val.append(misfit_2)
                    self.memo_loss_val.append(loss)
                    print(
                        f"VAL: step:{i}, l0:{misfit_0:.2e}, l1:{misfit_1:.2e}, l2:{misfit_2:.2e}, L:{loss:.2e}, lr:{self.learning_rate:.4e}"
                    )
                    self.memo_val_steps.append(i)
                    if not (
                        os.path.exists(
                            f"./models/models_{self.index_test_case}"
                        )
                    ):
                        os.makedirs(f"./models/models_{self.index_test_case}/")

                    self.model.save_weights(
                        f"./models/models_{self.index_test_case}/model_{i+1}/model_weights"
                    )
                    # self.show_curves(save=True, i=i + 1)
                    # self.show_results()

        except KeyboardInterrupt:
            pass

        # self.show_curves()

        if not (os.path.exists(f"./misfits_train/")):
            os.makedirs(f"./misfits_train/misfits_0")
            os.makedirs(f"./misfits_train/misfits_1")
            os.makedirs(f"./misfits_train/misfits_2")
            os.makedirs(f"./misfits_train/loss")
        if not (os.path.exists(f"./misfits_val/")):
            os.makedirs(f"./misfits_val/misfits_0")
            os.makedirs(f"./misfits_val/misfits_1")
            os.makedirs(f"./misfits_val/misfits_2")
            os.makedirs(f"./misfits_val/loss")

        np.save(
            f"./misfits_val/misfits_0/model_{self.index_test_case}.npy",
            np.array(self.memo_misfit_0_val),
        )
        np.save(
            f"./misfits_val/misfits_1/model_{self.index_test_case}.npy",
            np.array(self.memo_misfit_1_val),
        )
        np.save(
            f"./misfits_val/misfits_2/model_{self.index_test_case}.npy",
            np.array(self.memo_misfit_2_val),
        )
        np.save(
            f"./misfits_val/loss/model_{self.index_test_case}.npy",
            np.array(self.memo_loss_val),
        )
        np.save(
            f"./misfits_train/misfits_0/model_{self.index_test_case}.npy",
            np.array(self.memo_misfit_0_train),
        )
        np.save(
            f"./misfits_train/misfits_1/model_{self.index_test_case}.npy",
            np.array(self.memo_misfit_1_train),
        )
        np.save(
            f"./misfits_train/misfits_2/model_{self.index_test_case}.npy",
            np.array(self.memo_misfit_2_train),
        )
        np.save(
            f"./misfits_train/loss/model_{self.index_test_case}.npy",
            np.array(self.memo_loss_train),
        )

    def train_one_epoch(self, misfit_level, i):
        ti0 = time.time()
        batch_size = 64
        nb_data = len(self.data.X_train)
        nb_batch = nb_data // batch_size
        if nb_batch == 0:
            batch_size = nb_data
            nb_batch = 1
        if nb_batch <= 2:
            print(
                f"Warning, nb_data={nb_data} is small compared to batchsize={batch_size}, the number of batch per epoch is {nb_batch} "
            )

        rand_i = tf.random.shuffle(range(len(self.data.X_train)))
        X = tf.gather(self.data.X_train, rand_i)
        Y = tf.gather(self.data.Y_train, rand_i)

        misfit_0, misfit_1, misfit_2, loss = (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

        misfit_0, misfit_1, misfit_2, loss = self.train_one_epoch_misfit_acc(
            X, Y, batch_size, nb_batch, misfit_level
        )

        self.memo_misfit_0_train.append(misfit_0)
        self.memo_misfit_1_train.append(misfit_1)
        self.memo_misfit_2_train.append(misfit_2)
        self.memo_loss_train.append(loss)
        print(
            f"step:{i}, l0:{misfit_0:.2e}, l1:{misfit_1:.2e}, l2:{misfit_2:.2e}, L:{loss:.2e}, duration:{(time.time()-ti0):.2f}"
        )
