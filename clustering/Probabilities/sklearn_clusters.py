from sklearn.cluster import KMeans

def kmeans_probability(data, num_clusters):
    clustering = KMeans(n_clusters=num_clusters)
    X = data['prob_class1'].to_numpy().reshape(-1, 1)
    y = data['label'].to_numpy().reshape(-1, 1)
    clustering.fit(X)
    preds = clustering.labels_
    labels = y.flatten()
    for pred in sorted(set(preds)):
        split = labels[preds == pred]
        print(
            f'Cluster {pred}: F1 {2 * split.sum() / (split.shape[0] + labels.sum())}, P {split.sum() / split.shape[0]}, R {split.sum() / labels.sum()}')


def kmeans_logits(data, num_clusters):
    clustering = KMeans(n_clusters=num_clusters)
    X = data[['logit0', 'logit1']].to_numpy()
    y = data['label'].to_numpy().reshape(-1, 1)

    clustering.fit(X)
    preds = clustering.labels_
    labels = y.flatten()

    for pred in sorted(set(preds)):
        split = labels[preds == pred]
        print(
            f'Cluster {pred}: F1 {2 * split.sum() / (split.shape[0] + labels.sum())}, P {split.sum() / split.shape[0]}, R {split.sum() / labels.sum()}')





