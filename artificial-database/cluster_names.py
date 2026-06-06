CLUSTER_NAMES = {
    0: "Steady_Worker",
    1: "Balanced_Engager",
    2: "Low_Engagement",
    3: "Night_Crammer",
}


def name(cluster_id):
    return CLUSTER_NAMES.get(int(cluster_id), f"Cluster {cluster_id}")
