import os
import tarfile
import requests
from PIL import Image
import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split


def get_root():
    home = os.path.expanduser('~')
    return os.path.join(home, '.mps/mpsc20190813')


def create_root():
    if not os.path.exists(get_root()):
        os.makedirs(get_root())
        return True
    return False


def get_data_path(name):
    return os.path.join(get_root(), name)


def download_gz(url, name):
    print('Download:', name)
    response = requests.get(url)
    if response.status_code != 200:
        return False
    gz_name = url.split('/')[-1]

    print('Extract:', name)
    gz_path = os.path.join(get_root(), gz_name)
    with open(gz_path, 'bw') as f:
        f.write(response.content)
    with tarfile.open(gz_path, 'r:gz') as z:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(z, get_data_path(name))
    os.remove(gz_path)


def caltech101(width=150, height=150):
    if not os.path.exists(get_root()):
        create_root()
    if not os.path.exists(get_data_path('caltech101')):
        url = 'http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz'
        download_gz(url, 'caltech101')
    path = os.path.join(get_data_path('caltech101'), '101_ObjectCategories')
    categories = sorted(os.listdir(path))
    images = []
    labels = []
    category_names = []
    for l, c in enumerate(categories):
        category_path = os.path.join(path, c)
        for i in os.listdir(category_path):
            image = Image.open(os.path.join(category_path, i)).resize((width, height))
            image = np.asarray(image).astype("f")
            if len(image.shape) < 3:
                image = np.tile(image, 3).reshape(image.shape[0], image.shape[1], 3)
            elif image.shape[2] > 3:
                image = image[:, :, :3]
            images.append(image)
            category_names.append(c)
            labels.append(l)
    return np.array(images), np.array(labels), category_names


def artificial2d(n_clusters_for_each_class, n_data, min_scale=0.001, max_scale=0.1):
    means = np.random.uniform(-3, 3, (2, n_clusters_for_each_class, 2))
    scales = np.random.uniform(min_scale, max_scale, (2, n_clusters_for_each_class, 2))
    X, y = [], []
    for c, (class_means, class_scales) in enumerate(zip(means, scales)):
        for mean, scale in zip(class_means, class_scales):
            X = X + np.random.normal(mean, scale, (n_data, 2)).tolist()
            y = y + [c] * n_data
    return np.array(X), np.array(y)


def generate_fewshot_dataset(X, y, n_way, n_shot, n_pretrained_classes, r_test_queries=0.25, sparse=False):
    def renumber(y, mapping):
        return np.array([mapping[j] for j in y.ravel()])

    def shuffle(X, y):
        perm = np.random.permutation(len(X))
        return X[perm], y[perm]

    n_classes = len(np.unique(y))
    perm = np.random.permutation(n_classes)
    if n_pretrained_classes != 0:
        idx_pretrained = np.where(np.isin(y, perm[:n_pretrained_classes]))[0]
        X_pretrained = X[idx_pretrained]
        pretrained_label_mapping = {k: j for j, k in enumerate(np.unique(y[idx_pretrained]))}
        y_pretrained = renumber(y[idx_pretrained], pretrained_label_mapping)
        if not sparse:
            y_pretrained = np_utils.to_categorical(y_pretrained)
        pretrained = (X_pretrained, y_pretrained)
    else:
        pretrained = (None, None)

    n_episodes = int((n_classes - n_pretrained_classes) / n_way)
    episodes = []
    for i in range(n_episodes):
        classes = perm[n_pretrained_classes + i * n_way: n_pretrained_classes + (i + 1) * n_way]
        idx_episodes = []
        idx_episode_examples = []
        idx_episode_queries = []
        for cls in classes:
            idx_episode = np.random.permutation(np.where(y == cls)[0]).tolist()
            idx_episodes = idx_episodes + idx_episode
            idx_episode_examples = idx_episode_examples + idx_episode[:n_shot]
            idx_episode_queries = idx_episode_queries + idx_episode[n_shot:]
        episode_label_mapping = {k: j for j, k in enumerate(np.unique(y[idx_episodes]))}
        X_episode_examples, y_episode_examples = X[idx_episode_examples], y[idx_episode_examples]
        y_episode_examples = renumber(y_episode_examples, episode_label_mapping)
        X_episode_queries, y_episode_queries = shuffle(X[idx_episode_queries], y[idx_episode_queries])
        y_episode_queries = renumber(y_episode_queries, episode_label_mapping)
        if not sparse:
            y_episode_examples = np_utils.to_categorical(y_episode_examples)
            y_episode_queries = np_utils.to_categorical(y_episode_queries)
        X_episode_queries_train, X_episode_queries_test, y_episode_queries_train, y_episode_queries_test \
            = train_test_split(X_episode_queries, y_episode_queries, test_size=r_test_queries)
        episodes.append((X_episode_examples, y_episode_examples,
                         X_episode_queries_train, y_episode_queries_train,
                         X_episode_queries_test, y_episode_queries_test))
    return episodes, pretrained


if __name__ == '__main__':
    X, y, c = caltech101()
    print(X.shape, y.shape)
    print(len(c))
