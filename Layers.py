from __future__ import absolute_import, print_function, division

import tensorflow as tf
import numpy as np


def outer(x, y):
     return tf.matmul(tf.expand_dims(x, 1), tf.transpose(tf.expand_dims(y, 1)))


class Layer(object):
    def __init__(self, *args, **kwargs):
        self._build(*args, **kwargs)

    def __call__(self, x, *args, **kwargs):
        return self.call(x, *args, **kwargs)

    def call(self, x, *args, **kwargs):
        raise NotImplementedError('Not implemented in abstract class')


with tf.variable_scope('implict_hypernet') as vs:
    hypernet_vs = vs

with tf.variable_scope('') as vs:
    none_vs = vs


class BBHDiscriminator(object):
    def __init__(self, input_dim=1, units=[20, 20]):
        self.layers = []
        with tf.variable_scope(none_vs):
            with tf.variable_scope('implicit_discriminator'):
                for unit in units:
                    layer = tf.layers.Dense(unit, activation=tf.nn.relu)
                    layer.build((None, input_dim))
                    self.layers.append(layer)

                    input_dim = unit

                layer = tf.layers.Dense(1)
                layer.build((None, input_dim))
                self.layers.append(layer)

    def __call__(self, x, *args, **kwargs):
        for layer in self.layers:
            x = layer(x)

        return x
    

class BBHLayer(Layer):
    share_noise = True
    
    def _get_weight(self, name, size, units=[16, 32], use_bias=True,
                    noise_shape=1, num_samples=5, num_slices=1,
                    activation_func=lambda x: tf.maximum(0.1 * x, x)):
        slice_size = size[-1]
        assert slice_size % num_slices == 0

        gen_size = slice_size // num_slices

        cond = tf.eye(num_slices)

        with tf.variable_scope(hypernet_vs):
            with tf.variable_scope(name):
                flat_size = np.prod(size[:-1]) * gen_size
                
                if self.share_noise:
                    bbh_noise_col = tf.get_collection('bbh_noise')
                    if len(bbh_noise_col) == 1:
                        z = bbh_noise_col[0]
                    else:
                        z = tf.random_normal((num_samples, noise_shape))
                        tf.add_to_collection('bbh_noise', z)
                else:
                    z = tf.random_normal((num_samples, noise_shape))

                z = tf.stack([
                    tf.concat([
                        tf.tile(tf.expand_dims(z[s_dim], 0), [num_slices, 1]),
                        cond], 1) for s_dim in range(num_samples)])
                # [noise,cond, ..]

                for unit in units:
                    z = tf.layers.dense(inputs=z, units=unit, use_bias=use_bias)

                    z = activation_func(z)

                z = tf.layers.dense(inputs=z, units=flat_size,
                                    use_bias=use_bias)

                w = tf.reshape(z, [num_samples, -1])

                tf.add_to_collection('gen_weights', w)
                tf.add_to_collection('weight_samples', w)

return tf.reshape(w, [num_samples, ] + list(size))

class BBHDenseLayer(BBHLayer):
    def _build(self, name, input_dim, output_dim, use_bias=True,
               h_units=[16, 32],
               h_use_bias=True, h_noise_shape=1,
               num_samples=5, num_slices=1,
               aligned_noise=True,
               h_activation_func=lambda x: tf.maximum(0.1 * x, x)):
        self.share_noise = aligned_noise
        self.use_bias = use_bias
        with tf.variable_scope(name):
            self.w = self._get_weight(
                '{}/w'.format(name), (input_dim, output_dim), units=h_units,
                use_bias=h_use_bias, noise_shape=h_noise_shape,
                num_samples=num_samples, num_slices=num_slices,
                activation_func=h_activation_func)
            if self.use_bias:
                self.b = self._get_weight(
                    '{}/b'.format(name), (output_dim, ), units=h_units,
                    use_bias=h_use_bias, noise_shape=h_noise_shape,
                    num_samples=num_samples, num_slices=1,
                    activation_func=h_activation_func)

    def call(self, x, sample=0):
        x = tf.matmul(x, self.w[sample])

        if self.use_bias:
x = x + self.b[sample]