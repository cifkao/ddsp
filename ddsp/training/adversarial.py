from ddsp.training import nn
import gin
import numpy as np
import tensorflow.compat.v2 as tf

tfkl = tf.keras.layers

@gin.register
class MultiPhaseTrainer(ddsp.train_util.Trainer):

  @tf.function
  def step_fn(self, batch):
    """Per-Replica training step."""
    n = len(self.model.loss_groups)
    for i in range(n):
      if i == self.step % n:
        losses = self.model.loss_groups[i]
        variables = self.model.variable_groups[i]

    with tf.GradientTape() as tape:
      _ = self.model(batch, training=True)
      total_loss = tf.reduce_sum(losses)
    # Clip and apply gradients.
    grads = tape.gradient(total_loss, variables)
    grads, _ = tf.clip_by_global_norm(grads, self.grad_clip_norm)
    self.optimizer.apply_gradients(zip(grads, variables))
    return self.model.losses_dict


@gin.register
class AdversarialAutoencoder(ddsp.models.Autoencoder):

  def __init__(self, z_discriminator, d_weight, name='adversarial_autoencoder', **kwargs):
    super().__init__(name=name, **kwargs)
    self.z_discriminator = z_discriminator
    self.d_weight = d_weight
    self.bce = tf.keras.losses.BinaryCrossentropy(
        from_logits=True,
        reduction=tf.keras.losses.Reduction.NONE)

    self.loss_names.remove('total_loss')
    self.loss_names.extend(['d_loss', 'd_confusion', 'total_loss'])

  def call(self, features, training=False, phase=None):
    """Run the core of the network, get predictions and loss."""
    conditioning = self.encode(features, training=training)
    audio_gen = self.decode(conditioning, training=training)

    if training:
      update_autoencoder = (phase == 0)
      d_predictions, d_labels = self.discriminate_z(conditioning['z'])

      if update_autoencoder:
        self.add_losses(features['audio'], audio_gen)

        # Make all labels 0.5 to confuse the discriminator
        d_labels = tf.ones_like(d_labels, dtype=tf.float32) / 2
        self.add_loss(-self.d_weight * self.bce(d_predictions, d_labels))
      else:  # update discriminator
        self.add_loss(self.bce(d_predictions, d_labels))

    return audio_gen

  def discriminate_z(self, conditioning, training=False):
    if training:
      # Permute the global zs in half of the batch to get negative examples
      local_z, global_z = self.encoder.split_z(conditioning['z'])
      batch_size = local_z.shape[0]
      n_neg = batch_size // 2  # The number of negative examples to create
      indices = tf.concat([(tf.range(n_neg) + n_neg // 2) % n_neg,
                           tf.range(n_neg, batch_size)], axis=0)
      global_z = tf.gather(global_z, indices, axis=0)
      z = tf.concat(local_z, global_z, axis=-1)
      labels = tf.cast(tf.range(batch_size) >= n_neg), tf.int32)

    # Run the discriminator
    predictions = self.z_discriminator(z)
    return predictions, labels

  @property
  def losses_dict(self):
    """For metrics, returns dict {loss_name: loss_value}."""
    losses_dict = super().losses_dict
    losses_dict['total_loss'] = tf.reduce_sum(
        [v for k, v in losses_dict.items() if k != 'd_loss'])
    return losses_dict

  @property
  def loss_groups(self):
    return [
        [v for k, v in self.losses_dict if k not in ['d_loss', 'total_loss']],
        [self.losses_dict['d_loss']]
    ]

  @property
  def variable_groups(self):
    return [
        [self.encoder.trainable_variables + self.decoder_trainable_variables],
        [self.discriminator.trainable_variables]
    ]


@gin.register
class GlobalLocalZEncoder(MfccTimeDistributedRnnEncoder):

  def __init__(self, global_rnn_channels=512, global_rnn_type='gru',
               global_z_dims=32, name='global_local_z_encoder', **kwargs):
    super().__init__(name=name, **kwargs)

    self.global_rnn = nn.rnn(global_rnn_channels, global_rnn_type,
                             return_sequences=False)
    self.global_dense_out = nn.dense(global_z_dims)

  def call(self, conditioning):
    mfccs = spectral_ops.compute_mfcc(
        conditioning['audio'],
        lo_hz=20.0,
        hi_hz=8000.0,
        fft_size=self.fft_size,
        mel_bins=128,
        mfcc_bins=30,
        overlap=self.overlap,
        pad_end=True)
    conditioning['mfccs_normalized'] = self.z_norm(
        self.z_norm(mfccs[:, :, tf.newaxis, :])[:, :, 0, :])

    # Compute local z
    conditioning = super().call(conditioning)

    # Compute global z and concatenate to local z
    conditioning['global_z'] = self.compute_global_z(conditioning)
    time_steps = int(conditioning['z'].shape[1])
    conditioning['z'] = tf.concat(
        [conditioning['z'],
         self.expand_z(conditioning['global_z']], time_steps),
        axis=-1)

    return conditioning

  def compute_z(self, conditioning):
    # Run an RNN over the latents.
    z = self.rnn(conditioning['mfccs_normalized'])
    # Bounce down to compressed z dimensions.
    z = self.dense_out(z)
    return z

  def compute_global_z(conditioning):
    # Run an RNN over the latents.
    z = self.global_rnn(conditioning['mfccs_normalized'])
    # Bounce down to compressed z dimensions.
    z = self.global_dense_out(z)
    return z

  def split_z(z):
    return tf.split(z, [self.dense_out.units, self.global_dense_out.units], axis=-1)


@gin.register
class ZRnnDiscriminator(tfkl.Layer):

  def __init__(self,
               rnn_channels=512,
               rnn_type='gru',
               name='z_discriminator'):
    super().__init__(name=name)

    self.rnn = nn.rnn(rnn_channels, rnn_type, return_sequences=False)
    self.dense_out = nn.dense(1)

  def call(self, z):
    state = self.rnn(z)
    return self.dense_out(state)
