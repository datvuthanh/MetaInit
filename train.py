import os
import tensorflow as tf
import cProfile
import matplotlib.pyplot as plt

# Fetch and format the mnist data
(mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data()

dataset = tf.data.Dataset.from_tensor_slices(
  (tf.cast(mnist_images[...,tf.newaxis]/255, tf.float32),
   tf.cast(mnist_labels,tf.int64)))
dataset = dataset.shuffle(1000).batch(256)

# Build the model
mnist_model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(32,[3,3], activation='relu',
                         input_shape=(28, 28, 1)),
  tf.keras.layers.Conv2D(64,[3,3], activation='relu'),
  tf.keras.layers.Dropout(0.25),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128),
  tf.keras.layers.Dense(10)
])

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_history = []

eps = 1e-5
memory = [0] * len(mnist_model.trainable_variables)
momentum = 0.9
lr = 0.1
def train_step(images, labels,steps):
  with tf.GradientTape() as tape3:
    with tf.GradientTape() as tape2:
      with tf.GradientTape() as tape1:
        logits = mnist_model(images, training=True)

        # Add asserts to check the shape of the output.
        #tf.debugging.assert_equal(logits.shape, (32, 10))

        loss_value = loss_object(labels, logits)

      grad = tape1.gradient(loss_value, mnist_model.trainable_variables) # Gradient order 1
    
      #prod = tape2.gradient(grad,mnist_model.trainable_variables) # Gradient order 2
      hessian = tf.reduce_sum([tf.reduce_sum(g**2) / 2 for g in grad]) # Gradient order 2 with 1/2g^2
    prod = tape2.gradient(hessian,mnist_model.trainable_variables) # original hessian matrix of author

    firstIter = True
    for g,p in zip(grad,prod):
      g_new = 2 * tf.cast((g>= 0), dtype= tf.float32) - 1
      out = (g-p) / (g + eps * g_new) - 1
      out = tf.abs(out)
      out = tf.reduce_sum(out)
      if firstIter == True:
        total_out = out
        firstIter = False
      else:
        total_out = total_out + out 
          
    gq = total_out / mnist_model.count_params()

  if steps % 10 == 0:
    print("LOSS VALUE: ", gq.numpy())
  loss_history.append(gq.numpy().mean())
  final_grads = tape3.gradient(gq, mnist_model.trainable_variables)    
  
  #optimizer.apply_gradients(zip(final_grads, mnist_model.trainable_variables))

  for j, (p, g_all) in enumerate(zip(mnist_model.trainable_variables, final_grads)):
    norm = tf.norm(p)
    g = tf.sign(tf.reduce_sum(p * g_all) / norm)
    memory[j] = momentum * memory[j] - lr * g
    new_norm = norm + memory[j]
    p = tf.multiply(p,new_norm / norm)

def train(epochs):
  for epoch in range(epochs):
    for (batch, (images, labels)) in enumerate(dataset):
      train_step(images, labels,batch)
    print ('Epoch {} finished'.format(epoch))
    
train(epochs = 3)

plt.plot(loss_history)
plt.xlabel('Batch #')
plt.ylabel('Loss [entropy]')
