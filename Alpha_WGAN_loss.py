import tensorflow as tf

loss_object = tf.losses.BinaryCrossentropy(from_logits=True)


def d_f_loss(f_logit):
    return loss_object(tf.zeros_like(f_logit), f_logit)


def d_r_loss(r_logit):
    return loss_object(tf.ones_like(r_logit), r_logit)


def c_f_loss(f_logit):
    f_loss = loss_object(tf.ones_like(f_logit), f_logit)
    return f_loss


def c_r_loss(r_logit):
    f_loss = loss_object(tf.ones_like(r_logit), r_logit)
    return f_loss


def get_wgan_losses_fn():
    def d_loss_fn(r_logit, f_logit):
        r_loss = - tf.reduce_mean(r_logit)
        f_loss = tf.reduce_mean(f_logit)
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = - tf.reduce_mean(f_logit)
        return f_loss

    return d_loss_fn, g_loss_fn


def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))
    return loss


def gradient_penalty(discriminator, real, fake=None):
    def _interpolate(a, b=None):
        if b is None:  # interpolation in DRAGAN
            beta = tf.random.uniform(shape=tf.shape(a), minval=0., maxval=1.)
            b = a + 0.5 * tf.math.reduce_std(a) * beta
        shape = [tf.shape(a)[0]] + [1] * (a.shape.ndims - 1)
        alpha = tf.random.uniform(shape=shape, minval=0., maxval=1.)
        inter = a + alpha * (b - a)
        inter.set_shape(a.shape)
        return inter

    x = _interpolate(real, fake)
    with tf.GradientTape() as t:
        t.watch(x)
        pred = discriminator(x)
    grad = t.gradient(pred, x)
    norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
    gp = tf.reduce_mean((norm - 1.) ** 2)

    return gp
