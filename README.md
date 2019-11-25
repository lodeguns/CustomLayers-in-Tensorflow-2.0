# CustomLayers-in-Tensorflow-2.0
The script below shows how to modify, with a customized layer, the value of the tensors of a network being trained. This behaviour is modelled considering static and dynamic operation assignments. The layout is written with the new version of Tensorflow 2.0 and Keras 2.3.0 Cublas 10, Cudnn 7.

### Model and Execution

The toy problem is checked on MNIST.  

The model and the training is written in the attached script in python.
Run in the terminal: python3 customlayer.py

We have done the experiment on a **NVIDIA GeForce RTX 2060**.

**Environment:**
``` 
OS  - Ubuntu: 18.0
Tensorflow version: 2.0.0
Keras version: 2.3.0
Python version: 3.6.8
CUDA/cuDNN version:10.0/7.0
``` 

For those impatient, this is the modern layout,
with hidden placeholders according to the new version of TF.

```python

def create_model():
...
    model.add(CustomLayer())
...

#extract the i-th element from the tensor
def ith_element(x, i, j, n_chan):
    return tf.reshape( x[0, i:i + 1, j:j + 1, n_chan ], (-1,))

@tf.function
def dyn_assignment(z, row, col, sel_chan, T ):
    # some operatiions
    out_val0 = ith_element(z, row, col, sel_chan)*T
    out_val1 = ith_element(z, row, col, sel_chan)*T
    value_to_assign = tf.math.add_n([out_val0, out_val1, out_val1])
    z = assign_op_tensor(z, value_to_assign, row, col, sel_chan)
    return z

@tf.function
def static_assignment(x, row, col, sel_chan, T):
    up_val =  tf.constant([T], dtype=tf.float32)
    z = assign_op_tensor(x, up_val, row, col, sel_chan)
    return z

@tf.function
def assign_op_tensor(x, updates, cord_i, cord_j, n_chan):
    indices = tf.constant([[0, cord_i, cord_j, n_chan]])
    updated = tf.tensor_scatter_nd_update(x, indices, updates)
    return(updated)

@tf.function
def out_res(x):
    dim = x.shape
    h = dim[1]
    w = dim[2]
    n_chan = dim[3]
    row = 6
    col = 6
    sel_chan = 1
    z = tf.identity(x)
    z = dyn_assignment(z, row, col, sel_chan,  1/3)
    z = static_assignment(z, row, col, sel_chan, 1.)
    return z

@tf.custom_gradient
def custom_op(x):
    result = out_res(x) # do forward computation
    def custom_grad(dy):
        print(dy, [dy])
        grad = dy # compute gradient
        return grad
    return result, custom_grad

class CustomLayer(layers.Layer):
    def __init__(self):
        super(CustomLayer, self).__init__()

    def call(self, x):
        return custom_op(x)

```


Attached the issue and the problems faced to develop this: [solved issue](https://github.com/tensorflow/tensorflow/issues/34549)

Dott. Francesco Bardozzo (fbardozzo@unisa.it)

Dott. Mattia Delli Priscoli  (mdellipriscoli@unisa.it)

@NeuroneLab - University of Salerno - It




