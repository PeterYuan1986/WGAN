from tensorflow.python.data.experimental import prefetch_to_device
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE
from Alpha_WGAN_loss import *
from AlphaWGAN_tensorflow_RM import *
import argparse
import time
from Alpha_WGAN_dataset_preprocess import *
import tensorflow as tf


# ==============================================================================
# =                                   param                                    =
# ==============================================================================
def parse_args():
    desc = "Tensorflow implementation of Alpha_WGAN"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='train or test ?')
    parser.add_argument('--img_width', type=int, default='64', help='img_width')
    parser.add_argument('--img_height', type=int, default='64', help='img_height')
    parser.add_argument('--img_depth', type=int, default='64', help='img_depth')
    parser.add_argument('--grw', type=int, default='10', help='gradient_penalty_weight: Lamda1')
    parser.add_argument('--lamda2', type=int, default='10', help='Lamda2 in G_loss')
    parser.add_argument('--lr', type=int, default=0.0002, help='learning rate for all four model')
    parser.add_argument('--beta1', type=float, default=0.5, help='Decay rate for 1st moment of Adam')
    parser.add_argument('--latentdimension', type=int, default=1000, help='latent dimension')
    parser.add_argument('--iteration', type=int, default=10000, help='total iteration')
    parser.add_argument('--batch_size', type=int, default=1, help='The size of batch size')
    parser.add_argument('--g_iter', type=int, default=10, help='g_iter')
    parser.add_argument('--cd_iter', type=int, default=1, help='cd_iter')
    parser.add_argument('--d_iter', type=int, default=5, help='d_iter')
    parser.add_argument('--dataset', type=str, default='mri', help='dataset_name')
    parser.add_argument('--checkpoint_dir', type=str, default='model',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Directory name to save the samples on training')

    return parser.parse_args()


# ==============================================================================
# =                                    data                                    =
# ==============================================================================
def main():
    args = parse_args()
    checkpoint = './checkpoints'
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    checkpoint_dir = os.path.join(checkpoint, args.checkpoint_dir)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    dataset_name = args.dataset
    dataset_path = './dataset'
    if (args.phase == 'train'):
        datapath = os.path.join(dataset_path, dataset_name, 'train')
    else:
        datapath = os.path.join(dataset_path, dataset_name, 'test')

    img_class = Image_data(img_width=args.img_width, img_height=args.img_height, img_depth=args.img_depth,
                           dataset_path=datapath)
    img_class.preprocess()
    dataset = tf.data.Dataset.from_tensor_slices(img_class.dataset)
    dataset_num = len(img_class.dataset)  # all the images with different domain
    print("Dataset number : ", dataset_num)
    gpu_device = '/gpu:0'
    type(dataset)
    data_set = dataset.shuffle(buffer_size=dataset_num, reshuffle_each_iteration=True).repeat()
    data_set = data_set.batch(args.batch_size, drop_remainder=True)
    data_set = data_set.apply(prefetch_to_device(gpu_device, buffer_size=AUTOTUNE))
    data_set_iter = iter(data_set)

    # ==============================================================================
    # =                                   model                                    =
    # ==============================================================================

    G = Generator()
    D = Discriminator()
    C = Code_Discriminator()
    E = Encoder()

    G.summary()
    D.summary()
    C.summary()
    E.summary()

    G_optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, beta_1=args.beta1)
    D_optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, beta_1=args.beta1)
    C_optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, beta_1=args.beta1)
    E_optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, beta_1=args.beta1)

    """ Checkpoint """
    ckpt = tf.train.Checkpoint(G=G, D=D, C=C, E=E, G_optimizer=G_optimizer, D_optimizer=D_optimizer,
                               C_optimizer=C_optimizer, E_optimizer=E_optimizer)
    manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=1)

    if manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint).expect_partial()
        print('Latest checkpoint restored!!')
    else:
        print('Not restoring from saved checkpoint')

    g_iter = args.g_iter
    d_iter = args.d_iter
    cd_iter = args.cd_iter
    TOTAL_ITER = args.iteration
    start_time = time.time()

    for iteration in range(TOTAL_ITER):

        C.trainable = False
        D.trainable = False
        E.trainable = True
        G.trainable = True
        ###############################################
        # Train Encoder - Generator
        ###############################################

        for iters in range(g_iter):
            iter_start_time = time.time()
            with tf.GradientTape() as e, tf.GradientTape() as t:
                real_images = next(data_set_iter)
                z_rand = tf.random.normal(shape=(args.batch_size, args.latentdimension), mean=0, stddev=1)
                z_hat = E(real_images)
                x_hat = G(z_hat)  # x_rec
                x_rand = G(z_rand)
                c_loss = -tf.math.reduce_mean(C(z_hat))
                d_fake_loss = tf.math.reduce_mean(D(x_rand))
                d_real_loss = tf.math.reduce_mean(D(x_hat))
                d_loss = -d_fake_loss - d_real_loss
                l1_loss = args.lamda2 * L1_loss(x_rand, real_images)
                loss1 = d_loss + l1_loss + c_loss
            E_grad = e.gradient(loss1, E.trainable_variables)
            E_optimizer.apply_gradients(zip(E_grad, E.trainable_variables))
            G_grad = t.gradient(loss1, G.trainable_variables)
            G_optimizer.apply_gradients(zip(G_grad, G.trainable_variables))

        ###############################################
        # Train D
        ###############################################
        C.trainable = False
        D.trainable = True
        E.trainable = False
        G.trainable = False

        for iters in range(d_iter):
            iter_start_time = time.time()
            with tf.GradientTape() as a:
                z_rand = tf.random.normal(shape=(args.batch_size, args.latentdimension), mean=0, stddev=1)
                real_images = next(data_set_iter)
                z_hat = E(real_images)
                x_hat = G(z_hat)
                x_rand = G(z_rand)
                x_loss2 = -2 * tf.reduce_mean(D(real_images)) + tf.reduce_mean(D(x_rand))+ tf.reduce_mean(D(x_hat))

                gradient_penalty_r = gradient_penalty(D, real_images, x_rand)
                gradient_penalty_h = gradient_penalty(D, real_images, x_hat)
                loss2 = x_loss2 + gradient_penalty_r + gradient_penalty_h

                D_grad = a.gradient(loss2, D.trainable_variables)
                D_optimizer.apply_gradients(zip(D_grad, D.trainable_variables))

        ###############################################
        # Train CD
        ###############################################
        C.trainable = True
        D.trainable = False
        E.trainable = False
        G.trainable = False

        for iters in range(cd_iter):
            with tf.GradientTape() as t:
                z_rand = tf.random.normal(shape=(args.batch_size, args.latentdimension))

                gradient_penalty_cd = gradient_penalty(C, z_hat, z_rand)
                loss3 = -tf.reduce_mean(C(z_rand)) - c_loss + gradient_penalty_cd

                C_grad = t.gradient(loss3, C.trainable_variables)
                C_optimizer.apply_gradients(zip(C_grad, C.trainable_variables))

        print("iter: [%6d/%6d] time: %4.4f d_loss: %.8f, g_loss: %.8f, c_loss: %.8f" % (
            iteration, TOTAL_ITER, time.time() - iter_start_time, loss2, loss1, loss3))

        if iteration % 50 == 0:
            ckpt.save(file_prefix=checkpoint_prefix)

        if iteration % 50 == 0:
            sample = postprocess_images(x_rand)
            new_image = nib.Nifti1Image(np.int16(sample[0]), affine=np.eye(4))
            name = dataset_name + '_iteration_' + str(iteration) + '.nii'
            sample_dir = './sample'
            if not os.path.exists(sample_dir):
                os.makedirs(sample_dir)
            sample_dir = os.path.join(sample_dir, name)
            nib.save(new_image, sample_dir)
    print("trainning done. Total trainning time: %4.4f" % (time.time() - start_time))


if __name__ == '__main__':
    main()
