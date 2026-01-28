def extract_building(points, labels, eps=1.5, min_pts=200) -> clusters.labels_:
    building_pts = points[labels == BUILDING]
    clusters = DBSCAN(eps=eps).fit(building_pts[:, :2])
    return clusters.labels_
