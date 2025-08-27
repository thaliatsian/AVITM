#!/usr/bin/python

import warnings
import numpy as np
import tensorflow as tf
import time
import sys, os, getopt
import pickle
from models import prodlda, nvlda
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary

# Suppress warnings from TensorFlow
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ---------------- GPU Setup ---------------- #
def setup_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Detected {len(gpus)} GPU(s):")
        for gpu in gpus:
            print("  ", gpu)
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
    else:
        print("No GPU detected. Running on CPU.")

# ---------------- One-hot encoding ---------------- #
def onehot(data, min_length):
    return np.bincount(data, minlength=min_length)

# ---------------- Mini-batch generator ---------------- #
def create_minibatch(data, batch_size):
    rng = np.random.RandomState(10)
    while True:
        ixs = rng.randint(data.shape[0], size=batch_size)
        yield data[ixs]

# ---------------- Network Architecture ---------------- #
def make_network(layer1=100, layer2=100, num_topics=50, bs=200, eta=0.002, input_dim=None):
    tf.compat.v1.reset_default_graph()
    network_architecture = dict(
        n_hidden_recog_1=layer1,
        n_hidden_recog_2=layer2,
        n_hidden_gener_1=input_dim,
        n_input=input_dim,
        n_z=num_topics
    )
    return network_architecture, bs, eta

# ---------------- Training ---------------- #
def train(network_architecture, minibatches, model_type='prodlda',
          learning_rate=0.001, batch_size=200, training_epochs=100,
          prior='dirichlet', bn=False):
    tf.compat.v1.reset_default_graph()
    if model_type == 'prodlda':
        vae = prodlda.VAE(network_architecture,
                          learning_rate=learning_rate,
                          batch_size=batch_size,
                          prior=prior,
                          use_batch_norm=bn)
    elif model_type == 'nvlda':
        vae = nvlda.VAE(network_architecture,
                        learning_rate=learning_rate,
                        batch_size=batch_size)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    emb = None
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples_tr / batch_size)
        for i in range(total_batch):
            batch_xs = next(minibatches)
            cost, emb = vae.partial_fit(batch_xs)
            avg_cost += cost / n_samples_tr * batch_size

            if np.isnan(avg_cost):
                print(epoch, i, np.sum(batch_xs, 1).astype(int), batch_xs.shape)
                print('Encountered NaN, stopping training. Check learning rate or momentum.')
                sys.exit()

        if epoch % 5 == 0:
            print(f"Epoch: {epoch+1:04d} cost= {avg_cost:.9f}")
    return vae, emb

# ---------------- Print topics ---------------- #
def print_top_words(beta, feature_names, n_top_words=10):
    print('---------------Printing the Topics------------------')
    for i in range(len(beta)):
        print(" ".join([feature_names[j] for j in beta[i].argsort()[:-n_top_words - 1:-1]]))
    print('---------------End of Topics------------------')

# ---------------- Perplexity ---------------- #
def calcPerp(model):
    cost = []
    for doc in docs_te:
        doc_raw = doc.astype('float32')
        n_d = np.sum(doc_raw)
        if n_d == 0:
            continue
        c = model.test(doc_raw)
        cost.append(c / n_d)
    ppl = np.exp(np.mean(np.array(cost)))
    print('The approximated perplexity is: ', ppl)

# ---------------- Coherence ---------------- #
def compute_coherence(emb, top_n=10):
    global vocab, docs_tr
    if emb is None or len(emb) == 0:
        raise ValueError("Topic-word matrix (emb) is empty or None")

    emb_norm = emb / emb.sum(axis=1, keepdims=True)
    feature_names = list(zip(*sorted(vocab.items(), key=lambda x: x[1])))[0]

    docs = []
    for doc in docs_tr:
        words = []
        for idx in np.where(doc > 0)[0]:
            words.extend([feature_names[idx]] * int(doc[idx]))
        docs.append(words)

    dictionary = Dictionary(docs)
    topics = [[feature_names[i] for i in topic.argsort()[-top_n:][::-1]]
              for topic in emb_norm]

    cm = CoherenceModel(topics=topics, texts=docs, dictionary=dictionary, coherence="c_npmi")
    print("The c_npmi coherence score is:", cm.get_coherence())

# ---------------- Main ---------------- #
def main(argv):
    setup_gpu()

    dataset_tr = 'data/20news_clean/train.txt.npy'
    dataset_te = 'data/20news_clean/test.txt.npy'
    vocab_file = 'data/20news_clean/vocab.pkl'

    data_tr = np.load(dataset_tr, allow_pickle=True, encoding="latin1")
    data_te = np.load(dataset_te, allow_pickle=True, encoding="latin1")
    vocab_local = pickle.load(open(vocab_file, "rb"))
    vocab_size = len(vocab_local)

    print('Converting data to one-hot representation')
    data_tr = np.array([onehot(doc.astype('int'), vocab_size) for doc in data_tr if np.sum(doc) != 0])
    data_te = np.array([onehot(doc.astype('int'), vocab_size) for doc in data_te if np.sum(doc) != 0])
    print('Data Loaded')
    print('Dim Training Data', data_tr.shape)
    print('Dim Test Data', data_te.shape)

    global docs_tr, docs_te, vocab, n_samples_tr, n_samples_te
    docs_tr, docs_te = data_tr, data_te
    vocab = vocab_local
    n_samples_tr, n_samples_te = data_tr.shape[0], data_te.shape[0]

    # Default args
    m, f, s, t, b, r, e = '', 100, 100, 50, 200, 0.002, 100
    prior, bn = 'dirichlet', False

    try:
        opts, args = getopt.getopt(argv, "hpnm:f:s:t:b:r:e:",
                                   ["model=", "layer1=", "layer2=", "num_topics=",
                                    "batch_size=", "learning_rate=", "training_epochs=",
                                    "prior=", "bn="])
    except getopt.GetoptError:
        print('Usage: python run.py -m <model> -f <#units> -s <#units> -t <#topics> '
              '-b <batch_size> -r <learning_rate> -e <training_epochs> --prior=<dirichlet|gaussian> --bn=<True|False>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('Usage: python run.py -m <model> -f <#units> -s <#units> -t <#topics> '
                  '-b <batch_size> -r <learning_rate> -e <training_epochs> --prior=<dirichlet|gaussian> --bn=<True|False>')
            sys.exit()
        elif opt == '-p':
            print('Running default ProdLDA settings...')
            m, f, s, t, b, r, e = 'prodlda', 100, 100, 50, 200, 0.002, 100
        elif opt == '-n':
            print('Running default NVLDA settings...')
            m, f, s, t, b, r, e = 'nvlda', 100, 100, 50, 200, 0.005, 300
        elif opt == "-m": m = arg
        elif opt == "-f": f = int(arg)
        elif opt == "-s": s = int(arg)
        elif opt == "-t": t = int(arg)
        elif opt == "-b": b = int(arg)
        elif opt == "-r": r = float(arg)
        elif opt == "-e": e = int(arg)
        elif opt == "--prior": prior = arg.lower()
        elif opt == "--bn": bn = (arg.lower() == "true")

    minibatches = create_minibatch(docs_tr.astype('float32'), batch_size=b)
    network_architecture, batch_size, learning_rate = make_network(f, s, t, b, r, input_dim=data_tr.shape[1])
    print(network_architecture)
    print(opts)

    vae, emb = train(network_architecture, minibatches, m,
                     training_epochs=e, batch_size=batch_size,
                     learning_rate=learning_rate,
                     prior=prior, bn=bn)

    features = list(zip(*sorted(vocab.items(), key=lambda x: x[1])))[0]
    print_top_words(emb, features)

    print("The model is trained, now calculating the perplexity and coherence score...")
    compute_coherence(emb)
    calcPerp(vae)

if __name__ == "__main__":
    main(sys.argv[1:])
