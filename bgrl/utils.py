import random
import logging
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, pairwise
from sklearn import metrics


def set_random_seeds(random_seed=0):
    r"""Sets the seed for generating random numbers."""
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def get_logger(filename, verbosity=1, name=None, mode='a'):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    # formatter = logging.Formatter(
    #     "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    # )
    formatter = logging.Formatter(
        "%(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, mode)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def similarity_search(Z: torch.FloatTensor, Y: torch.LongTensor):
    """Evaluate node representations on node similarity search."""
    n_nodes = Z.shape[0]
    Z = Z.cpu()
    Y = Y.cpu()
    Z = torch.nn.functional.normalize(Z, dim=1)
    S = Z @ Z.T
    S.fill_diagonal_(-2.)
    S_ = torch.argsort(S, dim=1)
    Y_ = Y.repeat(n_nodes, 1)
    s = ""
    for N in [4, 8, 16, 5, 10, 20]:
        indices = S_[:, -N:]
        selected_label = Y_[torch.arange(n_nodes).repeat_interleave(N), indices.ravel()].view(n_nodes, N)
        original_label = Y.repeat_interleave(N).view(n_nodes, N)
        res = torch.mean(torch.sum((selected_label == original_label).float(), dim=1) / N) * 100
        s = s + f"Sim@{N}={res:.2f} "

    return s


def node_clustering(Z, Y):
    """Evaluate node representations on node clustering."""
    Z = torch.nn.functional.normalize(Z, dim=1)
    Z = Z.detach().cpu().numpy()
    nb_class = len(Y.unique())
    true_y = Y.detach().cpu().numpy()

    estimator = KMeans(n_clusters = nb_class, n_init=10)

    NMI_list = []
    h_list = []

    for i in range(10):
        estimator.fit(Z)
        y_pred = estimator.predict(Z)
        
        h_score = metrics.homogeneity_score(true_y, y_pred)
        s1 = normalized_mutual_info_score(true_y, y_pred, average_method='arithmetic')
        NMI_list.append(s1)
        h_list.append(h_score)

    NMI = np.array(NMI_list)
    Homo = np.array(h_list)
    clusterings = f'NMI={np.mean(NMI)*100:.2f}+-{np.std(NMI)*100:.2f} Homo={np.mean(Homo)*100:.2f}+-{np.std(Homo)*100:.2f}'
    return clusterings
    