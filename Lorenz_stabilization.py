import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Input
import matplotlib.pyplot as plt


def get_actor(x_shape):
    input_x = Input(shape=(x_shape,))
    x = layers.Dense(256, activation="sigmoid", name="layer1", kernel_initializer="RandomUniform", bias_initializer="RandomUniform")(input_x)
    x = layers.Dense(256, activation="sigmoid", name="layer2", kernel_initializer="RandomUniform", bias_initializer="RandomUniform")(x)
    x = layers.Dense(256, activation="sigmoid", name="layer3", kernel_initializer="RandomUniform", bias_initializer="RandomUniform")(x)
    output = layers.Dense(v_shape, activation=None)(x)
    actor = tf.keras.Model(inputs=input_x, outputs=output)
    return actor


def get_value(x_shape):
    input_x = Input(shape=(x_shape,))
    x = layers.Dense(256, activation="sigmoid", name="layer1", kernel_initializer="Zeros", bias_initializer="Zeros")(input_x)
    x = layers.Dense(256, activation="sigmoid", name="layer2", kernel_initializer="Zeros", bias_initializer="Zeros")(x)
    x = layers.Dense(256, activation="sigmoid", name="layer3", kernel_initializer="Zeros", bias_initializer="Zeros")(x)
    output = layers.Dense(1, activation="softplus")(x)
    critic = tf.keras.Model(inputs=input_x, outputs=output)
    return critic


def f(x, v):
    m1 = tf.constant([[1.], [0.], [0.]])
    m2 = tf.constant([[0.], [1.], [0.]])
    m3 = tf.constant([[0.], [0.], [1.]])
    x_1 = tf.matmul(x, m1)
    x_2 = tf.matmul(x, m2)
    x_3 = tf.matmul(x, m3)
    s = 1.
    b = 8./3.
    r = 2.
    f1 = s * (x_2 - x_1) + v
    f2 = x_1 * (r - x_3) - x_2
    f3 = x_1 * x_2 - b * x_3
    y = tf.matmul(f1, tf.transpose(m1)) + tf.matmul(f2, tf.transpose(m2)) + tf.matmul(f3, tf.transpose(m3))
    return y


def next_state(x, v, dt):
    y = x + f(x, v) * dt
    return y


def running_cost(x, v):
    x_cost = tf.matmul(x * x, tf.constant([[1.], [1.], [1.]]))
    L = 0.5 * x_cost * dt
    return L


def boundary_sampling(range, a):
    seed = tf.random.uniform(shape=(batch_size, x_shape), minval=-range, maxval=range)
    m1 = tf.constant([[a, 0., 0.]])
    A = tf.constant([[0., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    x = tf.matmul(tf.Variable(seed), A) + m1
    return x


def boundary_error(x1, x2):
    label1 = tf.zeros(shape=(batch_size, )) - 10.
    label2 = tf.zeros(shape=(batch_size,)) + 10.
    err1 = loss_function(actor(x1), label1)
    err2 = loss_function(actor(x2), label2)
    err = 0.5 * err1 + 0.5 * err2
    return err


def train_actor(x, i, alpha):
    with tf.GradientTape() as tape:
        v = actor(x, training=True)
        y = next_state(x, v, dt)
        running_cost_list = running_cost(x, v)
        Q_prime = critic(y)
        label = running_cost_list + gamma * Q_prime
        cost = tf.reduce_sum(label) / batch_size
        data = boundary_error(x1, x2)
        error = alpha * cost + (1. - alpha) * data
    if (i % 10 == 0):
        print("actor cost:", error)
    grad = tape.gradient(error, actor.trainable_weights)
    optimizer.apply_gradients(zip(grad, actor.trainable_weights))
    return None


def train_value(x, i):
    with tf.GradientTape() as tape:
        v = actor(x)
        y = next_state(x, v, dt)
        running_cost_list = running_cost(x, v)
        Q_prime = critic(y)
        label = running_cost_list + gamma * Q_prime
        logits = critic(x, training=True)
        error = loss_function(logits, label)
    if (i % 10 == 0):
        print("critic error:", error)
    grad = tape.gradient(error, critic.trainable_weights)
    optimizer.apply_gradients(zip(grad, critic.trainable_weights))
    return None


def path_generating(xx, n):
    v = actor(xx)
    control = []
    control.append(v)
    for i in range(1, n):
        xx = next_state(xx, v, dt)
        v = actor(xx)
    return xx


def sampling_state(range):
    seed = tf.random.uniform(shape=(batch_size, x_shape), minval=-range, maxval=range)
    x = tf.Variable(seed)
    return x


def graphing(xx, n):
    xxx = np.zeros(shape=(n, 1))
    yyy = np.zeros(shape=(n, 1))
    zzz = np.zeros(shape=(n, 1))
    xxx[0] = xx[0, 0].numpy()
    yyy[0] = xx[0, 1].numpy()
    zzz[0] = xx[0, 2].numpy()
    v = actor(xx)
    control = []
    control.append(v)
    for i in range(1, n):
        xx = next_state(xx, v, dt)
        v = actor(xx)
        control.append(v)
        xxx[i] = xx[0, 0].numpy()
        yyy[i] = xx[0, 1].numpy()
        zzz[i] = xx[0, 2].numpy()
    ax = plt.figure().add_subplot(projection='3d')
    xxx = np.reshape(xxx, newshape=(n, ))
    yyy = np.reshape(yyy, newshape=(n, ))
    zzz = np.reshape(zzz, newshape=(n, ))
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.plot(xxx, yyy, zzz, label='Trajectory', lw=1.3, linestyle='dashdot')
    ax.set_xlim(-1, 2)
    ax.set_ylim(-1, 2)
    ax.set_zlim(0, 2)
    ax.text(1., -1., 1., "Initial point")
    ax.text(0., 0., 0., "Target")
    plt.show()
    return [control, xxx, yyy, zzz]


x_shape = 3
v_shape = 1
batch_size = 1000
gamma = 0.9
dt = 0.01
loss_function = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
actor = get_actor(x_shape)
critic = get_value(x_shape)


for j in range(0, 6):
    x = sampling_state(1.)
    x1 = boundary_sampling(5., 1.)
    x2 = boundary_sampling(5., -1.)
    for i in range(0, 500):
        train_actor(x, i, 0.95)
    for i in range(0, 100):
        train_value(x, i)


xx = tf.Variable([[0.6, -0.9, 0.8]])
[control, xxx, yyy, zzz] = graphing(xx, 1500)
