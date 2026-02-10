import numpy as np
from sklearn.cluster import DBSCAN

BUILDING = 2


def extract_building(
    points: np.ndarray, labels: np.ndarray, eps: float = 1.5, min_pts: int = 200
) -> np.ndarray:
    building_pts = points[labels == BUILDING]
    clusters = DBSCAN(eps=eps, min_samples=min_pts).fit(building_pts[:, :2])
    return clusters.labels_
