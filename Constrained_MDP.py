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
    m1 = tf.constant([[1.], [0.]])
    m2 = tf.constant([[0.], [1.]])
    x_1 = tf.matmul(x, m1)
    x_2 = tf.matmul(x, m2)
    d = 0.6
    f1 = -d * x_1 - x_2
    f2 = x_1 ** 3 + x_2 * v
    y = tf.matmul(f1, tf.transpose(m1)) + tf.matmul(f2, tf.transpose(m2))
    return y


def next_state(x, v, dt):
    y = x + f(x, v) * dt
    return y


def running_cost(x, v):
    x_cost = tf.matmul(x * x, tf.constant([[1.], [1.]]))
    L = 0.5 * x_cost * dt
    return L


def boundary_sampling(range):
    seed = tf.random.uniform(shape=(batch_size2, 1), minval=-range, maxval=range)
    return seed


def boundary_error(x2):
    x1 = 1. - x2 ** 2
    x = tf.matmul(tf.Variable(x2), tf.constant([[0., 1.]])) + tf.matmul(tf.Variable(x1), tf.constant([[1., 0.]]))
    y = tf.math.sign(x2)
    label = 1./(x2) * (- x1**3 - 2. * y)
    err = loss_function(actor(x), label)
    return err


def train_actor(x, i, alpha):
    with tf.GradientTape() as tape:
        v = actor(x, training=True)
        y = next_state(x, v, dt)
        running_cost_list = running_cost(x, v)
        Q_prime = critic(y)
        label = running_cost_list + gamma * Q_prime
        cost = tf.reduce_sum(label) / batch_size1
        data = boundary_error(x2)
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


def plotting(N, xx):
    xxx = np.zeros(shape=(N, 1))
    yyy = np.zeros(shape=(N, 1))
    xxx[0] = xx[0, 0].numpy()
    yyy[0] = xx[0, 1].numpy()
    v = actor(xx)
    control = []
    control.append(v)
    for i in range(1, N):
        xx = next_state(xx, v, dt)
        v = actor(xx)
        control.append(v)
        xxx[i] = xx[0, 0].numpy()
        yyy[i] = xx[0, 1].numpy()
    plt.plot(xxx, yyy)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True)
    plt.show()
    return None


def sampling_state(l1, u1, l2, u2):
    x1 = tf.random.uniform(shape=(batch_size1, 1), minval=l1, maxval=u1)
    x2 = tf.random.uniform(shape=(batch_size1, 1), minval=l2, maxval=u2)
    x1 = (1. - x2**2 / 9.) * x1 - 8. * (x2**2) / 9.
    x = tf.matmul(tf.Variable(x1), tf.constant([[1., 0.]])) + tf.matmul(tf.Variable(x2), tf.constant([[0., 1.]]))
    x = tf.Variable(x)
    return x


x_shape = 2
v_shape = 1
batch_size1 = 1000
batch_size2 = 500
gamma = 0.9
dt = 0.01
loss_function = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Adam(learning_rate=0.001)

actor = get_actor(x_shape)
critic = get_value(x_shape)


for j in range(0, 6):
    x = sampling_state(-8., 1., -3., 3.)
    x2 = boundary_sampling(3.)
    for i in range(0, 1000):
        train_actor(x, i, 0.5)
    for i in range(0, 300):
        train_value(x, i)

xx = tf.Variable([[-4., 1.]])
plotting(1000, xx)


N = 1000
t = np.linspace(-2.5, 2.5, 501)
dt = 0.01
data1 = 1.- t**2
xx = tf.Variable([[-4., 1.]])
xc = np.zeros(shape=(N, 1))
yc = np.zeros(shape=(N, 1))
xc[0] = xx[0, 0].numpy()
yc[0] = xx[0, 1].numpy()
v = actor(xx)
for i in range(1, N):
    xx = next_state(xx, v, dt)
    v = actor(xx)
    xc[i] = xx[0, 0].numpy()
    yc[i] = xx[0, 1].numpy()


fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.plot(data1, t, color=color, label='Constraint')
ax1.plot(xc, yc, color='tab:blue',linestyle='dashdot', label='BVFM')
ax1.legend()
ax1.annotate('Admissible Set D', xy=(90, 200), xycoords='figure points')
ax1.set_xlim(-4, 1)
ax1.set_ylim(-2.5, 2.5)
ax1.grid()
plt.show()