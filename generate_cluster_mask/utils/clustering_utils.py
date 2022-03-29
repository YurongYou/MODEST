import numpy as np
import sklearn
import sklearn.neighbors as neighbors
import scipy
from .pointcloud_utils import estimate_plane, distance_to_plane

def precompute_affinity_matrix(
    ptc,
    pp_score,
    neighbor_type='mutual_knn',
    affinity_type='l1',
    n_neighbors=50,
    radius=1.,
):
    assert ptc.shape[0] == pp_score.shape[0]
    if neighbor_type == 'knn':
        graph = neighbors.kneighbors_graph(
            ptc[:, :3],  n_neighbors=n_neighbors, n_jobs=-1)
    elif neighbor_type == 'sym_knn':
        graph = neighbors.kneighbors_graph(
            ptc[:, :3],  n_neighbors=n_neighbors, n_jobs=-1)
        graph = graph + graph.T # scale does not matter here
        graph.eliminate_zeros()
    elif neighbor_type == 'mutual_knn':
        graph = neighbors.kneighbors_graph(
            ptc[:, :3],  n_neighbors=n_neighbors, n_jobs=-1)
        graph = graph.multiply(graph.T)  # scale does not matter here
        graph.eliminate_zeros()
    elif neighbor_type == 'radius':
        graph = neighbors.radius_neighbors_graph(
            ptc[:, :3], radius=radius, n_jobs=-1)
    elif neighbor_type == 'radius_mutual_knn':
        graph = neighbors.kneighbors_graph(
            ptc[:, :3], n_neighbors=n_neighbors, n_jobs=-1)
        graph = graph.multiply(graph.T)
        graph = graph.multiply(neighbors.radius_neighbors_graph(
            ptc[:, :3], radius=radius, n_jobs=-1))
        graph.eliminate_zeros()
    else:
        raise NotImplementedError(neighbor_type)

    dist_data = graph.data.copy()
    for _r in range(graph.indptr.shape[0]-1):
        if affinity_type == 'l1':
            dist_data[graph.indptr[_r]:graph.indptr[_r+1]] = \
                np.abs(
                    pp_score[_r] -
                    pp_score[graph.indices[graph.indptr[_r]:graph.indptr[_r+1]]])
        elif affinity_type == 'exp':
            dist_data[graph.indptr[_r]:graph.indptr[_r+1]] = \
                np.exp((pp_score[_r] -
                        pp_score[graph.indices[graph.indptr[_r]:graph.indptr[_r+1]]])**2)
        elif affinity_type == '3d_l2_distance':
            dist_data[graph.indptr[_r]:graph.indptr[_r+1]] = \
                np.linalg.norm(ptc[_r].reshape(1, -1) -
                               ptc[graph.indices[graph.indptr[_r]:graph.indptr[_r+1]]], axis=1)
        else:
            raise NotImplementedError(affinity_type)
    return scipy.sparse.csr_matrix(
        (dist_data, graph.indices, graph.indptr), shape=graph.shape)


def smoothing(
    ptc,
    pp_score,
    neighbor_type='knn',
    n_neighbors=50,
    radius=1.,
    num_iterations=10
):
    assert ptc.shape[0] == pp_score.shape[0]
    if neighbor_type == 'knn':
        graph = neighbors.kneighbors_graph(
            ptc[:, :3],  n_neighbors=n_neighbors,
            mode='distance',
            n_jobs=-1)
    elif neighbor_type == 'radius':
        graph = neighbors.radius_neighbors_graph(
            ptc[:, :3], radius=radius,
            mode='distance',
            n_jobs=-1)
    else:
        raise NotImplementedError(neighbor_type)
    graph.data = np.exp(-graph.data ** 2 / 2)
    graph.data = graph.data / \
        np.repeat(np.add.reduceat(graph.data, graph.indptr[:-1]),
                  np.diff(graph.indptr))
    pp_score = pp_score.copy()
    for _i in range(num_iterations):
        pp_score = np.asarray(graph.dot(
            pp_score.reshape(-1, 1))).squeeze()
    return pp_score

def is_valid_cluster(
        ptc,
        pp_score,
        plane,
        min_points=10,
        max_volume=40,
        min_volume=0.5,
        max_min_height=4,
        min_max_height=0,
        percentile=10,
        min_percentile_pp_score=0.7):
    if ptc.shape[0] < min_points:
        return False
    # volume = np.prod(ptc.max(axis=0) - ptc.min(axis=0))
    # if volume > max_volume or volume < min_volume:
    #     return False
    distance_to_ground = distance_to_plane(ptc, plane, directional=True)
    if distance_to_ground.min() > max_min_height:
        return False
    if distance_to_ground.max() < min_max_height:
        return False
    if np.percentile(pp_score, percentile) > min_percentile_pp_score:
        return False
    return True

def filter_labels(
    ptc,
    pp_score,
    labels,
    **kwargs
):
    labels = labels.copy()
    plane = estimate_plane(ptc, max_hs=-1.5, ptc_range=((-70, 70), (-50, 50)))
    for i in range(labels.max()+1):
        if not is_valid_cluster(
                ptc[labels == i, :3], pp_score[labels == i], plane, **kwargs):
            labels[labels == i] = -1
    label_mapping = sorted(list(set(labels)))
    label_mapping = {x:i for i, x in enumerate(label_mapping)}
    for i in range(len(labels)):
        labels[i] = label_mapping[labels[i]]
    return labels
