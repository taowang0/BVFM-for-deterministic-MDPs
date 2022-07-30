import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Input
import matplotlib.pyplot as plt
import time


def f(x, v):
    A = tf.constant([[-0.0665, 11.5, 0., 0.], [0., -2.5, 2.5, 0.], [-9.5, 0., -13.736, -13.736], [0.6, 0., 0., 0.]])
    B = tf.constant([[0., 0., 13.736, 0.]])
    y = tf.matmul(x, tf.transpose(A)) + tf.matmul(v, B)
    return y


def next_state(x, v, dt):
    y = x + f(x, v)*dt
    return y


def running_cost(x, v):
    x_cost = tf.matmul(x * x, tf.constant([[1.], [1.], [1.], [1.]]))
    v_cost = tf.matmul(v * v, tf.constant([[1.]]))
    L = 0.5 * (x_cost + v_cost) * dt
    return L


def get_actor(x_shape):
    input_x = Input(shape=(x_shape,))
    x = layers.Dense(256, activation="sigmoid", name="layer1", kernel_initializer="RandomUniform", bias_initializer="RandomUniform")(input_x)
    x = layers.Dense(256, activation="sigmoid", name="layer2", kernel_initializer="RandomUniform", bias_initializer="RandomUniform")(x)
    x = layers.Dense(256, activation="sigmoid", name="layer3", kernel_initializer="RandomUniform", bias_initializer="RandomUniform")(x)
    output = layers.Dense(v_shape, activation=None)(x)
    actor = tf.keras.Model(inputs=input_x, outputs=output)
    return actor


def get_critic(x_shape):
    input_x = Input(shape=(x_shape,))
    x = layers.Dense(256, activation="sigmoid", name="layer1", kernel_initializer="Zeros", bias_initializer="Zeros")(input_x)
    x = layers.Dense(256, activation="sigmoid", name="layer2", kernel_initializer="Zeros", bias_initializer="Zeros")(x)
    x = layers.Dense(256, activation="sigmoid", name="layer3", kernel_initializer="Zeros", bias_initializer="Zeros")(x)
    output = layers.Dense(1, activation="softplus")(x)
    critic = tf.keras.Model(inputs=input_x, outputs=output)
    return critic


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


def train_actor(x, i, alpha):
    with tf.GradientTape() as tape:
        v = actor(x, training=True)
        y = next_state(x, v, dt)
        running_cost_list = running_cost(x, v)
        Q_prime = critic(y)
        label = running_cost_list + gamma * Q_prime
        cost = tf.reduce_sum(label) / batch_size
        data = boundary_error(x1, 1.)
        error = alpha * cost + (1. - alpha) * data
    if (i % 10 == 0):
        print("actor cost:", error)
    grad = tape.gradient(error, actor.trainable_weights)
    optimizer.apply_gradients(zip(grad, actor.trainable_weights))
    return None


def boundary_sampling(range):
    seed = tf.random.uniform(shape=(batch_size2, x_shape), minval=-range, maxval=range)
    return seed


def boundary_error(xx, range):
    m1 = tf.constant([[0., 0., range, 0.]])
    A1 = tf.constant([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 1.]])
    x1 = tf.matmul(tf.Variable(xx), A1) + m1
    x2 = -x1
    label1 = tf.zeros(shape=(batch_size2, 1)) - 0.5*tf.matmul(x1 * x1, tf.constant([[1.], [1.], [1.], [1.]]))
    label2 = tf.zeros(shape=(batch_size2, 1)) + 0.5*tf.matmul(x2 * x2, tf.constant([[1.], [1.], [1.], [1.]]))
    err1 = loss_function(actor(x1), label1)
    err2 = loss_function(actor(x2), label2)
    err = 0.5 * (err1 + err2)
    return err


def sampling_state(range):
    seed = tf.random.uniform(shape=(batch_size, x_shape), minval=-range, maxval=range)
    x = tf.Variable(seed)
    return x


x_shape = 4
v_shape = 1
batch_size = 1000
batch_size2 = 250
dt = 0.05
gamma = 0.9
loss_function = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
actor = get_actor(x_shape)
critic = get_critic(x_shape)


for j in range(0, 1):
    x = sampling_state(1.)
    x1 = boundary_sampling(1.)
    for i in range(0, 500):
        train_actor(x, i, 0.85)
    for i in range(0, 100):
        train_value(x, i)


N = 320
xx = tf.Variable([[0.4, 0.4, -0.3, 0.3]])
xx1 = np.zeros(shape=(N, 1))
xx2 = np.zeros(shape=(N, 1))
xx3 = np.zeros(shape=(N, 1))
xx4 = np.zeros(shape=(N, 1))

t1 = np.arange(0.0, N*dt, dt)
xx1[0] = xx[0, 0].numpy()
xx2[0] = xx[0, 1].numpy()
xx3[0] = xx[0, 2].numpy()
xx4[0] = xx[0, 3].numpy()
v = actor(xx)
control = []
control.append(v)
for i in range(1, N):
    xx = next_state(xx, v, dt)
    v = actor(xx)
    control.append(v)
    xx1[i] = xx[0, 0].numpy()
    xx2[i] = xx[0, 1].numpy()
    xx3[i] = xx[0, 2].numpy()
    xx4[i] = xx[0, 3].numpy()
fig, ax = plt.subplots()

ax.plot(t1, xx1, color='blue', linestyle='solid')
ax.plot(t1, xx2, color='green', linestyle='dotted')
ax.plot(t1, xx3, color='red', linestyle='dashed')
ax.plot(t1, xx4, color='black', linestyle='dashdot')
ax.set_title('States')
ax.set_xlabel('time')
plt.legend(('x1', 'x2', 'x3', 'x4'), shadow=True)
plt.show()
ax.set_xlim(0, 16)
tb = time.time()

control1 = np.zeros(shape=(N, 1))
for i in range(0, N):
    control1[i] = control[i].numpy()

fig, ax = plt.subplots()
ax.set_title('Control Input')
ax.set_xlabel('Time t')
ax.set_ylabel('u(t)')
ax.plot(t1, control1, label='Trajectory', lw=1.3, linestyle='solid')
ax.set_xlim(0, 16)
ax.grid(True)
plt.show()
