import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from dataset import caltech101, generate_fewshot_dataset


if __name__ == '__main__':
    tf.enable_eager_execution()

    print('LOAD Clatech101')
    X, y, c = caltech101(150, 150)

    print('LOAD EPISODES')
    n_way = 10
    n_shot = 3
    episodes, pretrained = generate_fewshot_dataset(X, y, n_way, n_shot, 0)

    features = keras.models.Sequential([
        ResNet50(include_top=False, input_shape=(150, 150, 3), classes=10),
        # VGG16(include_top=False, input_shape=(150, 150, 3), classes=10),
        keras.layers.Flatten()
    ])

    l = tf.Variable(0.01, name='l')
    alpha = tf.Variable(1e-4, name='alpha')
    beta = tf.Variable(1., name='beta')

    variables = features.trainable_variables + [l, alpha, beta]

    opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)

    datagen = ImageDataGenerator()
    epoch = 1
    batch_size = 64
    global_step = tf.Variable(0)

    episodes_history = []
    for epc, epoch in enumerate(range(epoch)):
        print('Epoch', epc)
        for eps, episode in enumerate(episodes):
            print('======================================================')
            print('Episode', eps)
            print('======================================================')
            X_example, y_example, X_query_train, y_query_train, X_query_test, y_query_test = episode

            X_example = tf.constant(X_example, dtype=tf.float32)
            y_example = tf.constant(y_example, dtype=tf.float32)

            n_batch = 1
            datagen.fit(X_query_train)
            for X_batch, y_batch in datagen.flow(X_query_train, y_query_train, batch_size=batch_size):
                with tf.GradientTape() as t:
                    Z = features(X_example)
                    ZT = tf.transpose(Z)
                    try:
                        ZZTlI_inv = tf.linalg.inv(tf.matmul(Z, ZT) + l * tf.eye(n_way * n_shot))
                    except:
                        print('CANNOT TAKE INV')
                        continue
                    W = tf.matmul(tf.matmul(ZT, ZZTlI_inv), y_example)

                    X_batch = tf.Variable(X_batch, dtype=tf.float32)
                    y_batch = tf.Variable(y_batch, dtype=tf.float32)

                    Z = features(X_batch)
                    y_hat = tf.clip_by_value(alpha * K.dot(Z, W) + beta, 1e-13, 1 - 1e-13)
                    current_loss = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_batch, logits=y_hat))
                grads = t.gradient(current_loss, variables)
                opt.apply_gradients(zip(grads, variables), global_step)
                print('batch', n_batch)
                print('acc:',
                      tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_hat, axis=1),
                                                      tf.argmax(y_batch, axis=1)), dtype=tf.float32)))
                Z = features(X_query_test)
                y_hat = tf.clip_by_value(alpha * K.dot(Z, W) + beta, 1e-13, 1 - 1e-13)
                print('val_acc',
                      tf.reduce_mean(
                          tf.cast(tf.equal(tf.argmax(y_hat, axis=1),
                                           tf.argmax(y_query_test, axis=1)), dtype=tf.float32)))
                n_batch = n_batch + 1
                if n_batch > 80:
                    break
