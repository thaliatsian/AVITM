import numpy as np
import tensorflow as tf

# Disable eager execution for TF1-style code
tf.compat.v1.disable_eager_execution()
tf.compat.v1.reset_default_graph()

def xavier_init(fan_in, fan_out, constant=1):
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.compat.v1.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)

class VAE(object):
    """
    Variational Autoencoder adapted for topic modeling (ProdLDA-like).
    """

    def __init__(self, network_architecture, transfer_fct=tf.nn.softplus,
                 learning_rate=0.001, batch_size=100):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        print('Learning Rate:', self.learning_rate)

        # tf Graph input
        self.x = tf.compat.v1.placeholder(tf.float32, [None, network_architecture["n_input"]])
        self.keep_prob = tf.compat.v1.placeholder(tf.float32)

        # Latent dimension
        self.h_dim = int(network_architecture["n_z"])
        self.a = np.ones((1, self.h_dim), dtype=np.float32)
        self.mu2 = tf.constant((np.log(self.a).T - np.mean(np.log(self.a), 1)).T)
        self.var2 = tf.constant((( (1.0/self.a)*(1 - (2.0/self.h_dim)) ).T +
                                (1.0/(self.h_dim*self.h_dim))*np.sum(1.0/self.a, 1)).T)

        self._create_network()
        self._create_loss_optimizer()

        init = tf.compat.v1.global_variables_initializer()
        self.sess = tf.compat.v1.InteractiveSession()
        self.sess.run(init)

    def _create_network(self):
        self.network_weights = self._initialize_weights(**self.network_architecture)
        self.z_mean, self.z_log_sigma_sq = \
            self._recognition_network(self.network_weights["weights_recog"],
                                      self.network_weights["biases_recog"])

        n_z = self.network_architecture["n_z"]
        eps = tf.compat.v1.random_normal((tf.shape(self.x)[0], n_z), 0, 1, dtype=tf.float32)
        self.z = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))
        self.sigma = tf.exp(self.z_log_sigma_sq)

        self.x_reconstr_mean = self._generator_network(self.z, self.network_weights["weights_gener"])
        print(self.x_reconstr_mean)

    def _initialize_weights(self, n_hidden_recog_1, n_hidden_recog_2,
                            n_hidden_gener_1, n_input, n_z):
        all_weights = dict()
        all_weights['weights_recog'] = {
            'h1': tf.compat.v1.get_variable('h1', [n_input, n_hidden_recog_1]),
            'h2': tf.compat.v1.get_variable('h2', [n_hidden_recog_1, n_hidden_recog_2]),
            'out_mean': tf.compat.v1.get_variable('out_mean', [n_hidden_recog_2, n_z]),
            'out_log_sigma': tf.compat.v1.get_variable('out_log_sigma', [n_hidden_recog_2, n_z])
        }
        all_weights['biases_recog'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))
        }
        all_weights['weights_gener'] = {
            'h2': tf.Variable(xavier_init(n_z, n_hidden_gener_1))
        }
        return all_weights

    def _recognition_network(self, weights, biases):
        # Probabilistic encoder (recognition network)
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.x, weights['h1']), biases['b1']))
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
        layer_do = tf.nn.dropout(layer_2, self.keep_prob)

        bn_mean = tf.keras.layers.BatchNormalization()
        z_mean = bn_mean(tf.add(tf.matmul(layer_do, weights['out_mean']), biases['out_mean']))
        bn_log_sigma = tf.keras.layers.BatchNormalization()
        z_log_sigma_sq = bn_log_sigma(tf.add(tf.matmul(layer_do, weights['out_log_sigma']),
                                             biases['out_log_sigma']))
        return z_mean, z_log_sigma_sq

    def _generator_network(self, z, weights):
        self.layer_do_0 = tf.nn.dropout(tf.nn.softmax(z), self.keep_prob)
        bn_gen = tf.keras.layers.BatchNormalization()
        x_reconstr_mean = tf.nn.softmax(bn_gen(tf.add(tf.matmul(self.layer_do_0, weights['h2']), 0.0)))
        return x_reconstr_mean

    def _create_loss_optimizer(self):
        eps = 1e-8  # numerical stability constant

        # ----- Reconstruction loss (cross-entropy) -----
        x_reconstr_mean_safe = tf.clip_by_value(self.x_reconstr_mean, eps, 1.0)
        reconstr_loss = -tf.reduce_sum(self.x * tf.math.log(x_reconstr_mean_safe), axis=1)

        # ----- Latent loss (KL divergence) -----
        sigma_safe = tf.clip_by_value(self.sigma, eps, 1e8)
        var2_safe = tf.clip_by_value(self.var2, eps, 1e8)

        latent_loss = 0.5 * (
            tf.reduce_sum(sigma_safe / var2_safe, axis=1)
            + tf.reduce_sum(tf.square(self.mu2 - self.z_mean) / var2_safe, axis=1)
            - self.h_dim
            + tf.reduce_sum(tf.math.log(var2_safe), axis=1)
            - tf.reduce_sum(self.z_log_sigma_sq, axis=1)
        )

        # ----- Total cost -----
        self.cost = tf.reduce_mean(reconstr_loss + latent_loss)

        # ----- Optimizer -----
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9)
        grads_vars = optimizer.compute_gradients(self.cost)
        grads, vars_ = zip(*grads_vars)

        # Replace NaN/Inf gradients with zeros
        safe_grads = [tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)) for g in grads]

        # Clip gradients by global norm
        clipped_grads, _ = tf.clip_by_global_norm(safe_grads, 5.0)

        # Apply gradients
        self.optimizer = optimizer.apply_gradients(zip(clipped_grads, vars_))

    def _normalize_input(self, X):
        """Normalize input for stable training (row sum = 1)"""
        X = np.clip(X, 0, 500)  # avoid extreme values
        row_sums = np.maximum(1.0, X.sum(axis=1, keepdims=True))
        return X / row_sums

    def partial_fit(self, X):
        """Run one training step and return cost + topic-word matrix"""
        X_norm = self._normalize_input(X)
        opt, cost, emb = self.sess.run(
            (self.optimizer, self.cost, self.network_weights['weights_gener']['h2']),
            feed_dict={self.x: X_norm, self.keep_prob: 0.4}
        )
        return cost, emb
    
    def test(self, X):
        """
        Compute the reconstruction + KL loss on a single document
        using raw word counts (no normalization).
        Used for perplexity evaluation.
        """
        X_raw = np.clip(X, 0, 500)  # keep counts safe
        cost = self.sess.run(
        self.cost,
        feed_dict={self.x: np.expand_dims(X_raw, axis=0), self.keep_prob: 1.0}
        )
        return cost