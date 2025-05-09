import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from ovito.io import import_file, export_file
from ovito.modifiers import PythonScriptModifier

min_samples = 5
percentiles = np.arange(10, 100, 5)

input_dir  = "out/dump"
csv_dir    = "out/csv"
os.makedirs(csv_dir, exist_ok=True)

for i in range(1, 10):
    file_name = f"key_area_{i}.dump"
    file_path = os.path.join(input_dir, file_name)
    if not os.path.isfile(file_path):
        print(f"⚠️ No existe {file_name}, saltando.")
        continue

    pipeline = import_file(file_path)
    data     = pipeline.compute()
    ids      = data.particles['Particle Identifier'].array
    types    = data.particles['Particle Type'].array
    coords   = data.particles.positions.array   

    nbrs, _ = NearestNeighbors(n_neighbors=min_samples).fit(coords).kneighbors(coords)
    k_dist  = np.sort(nbrs[:, -1])
    plt.figure()
    plt.plot(k_dist)
    plt.xlabel("Índice de punto (ordenado)")
    plt.ylabel(f"{min_samples}-ésima distancia")
    plt.title(f"k-distance para key_area_{i}")
    plt.show()

    best = {"sil": -1.0, "eps": None, "labels": None}
    for p in percentiles:
        eps_cand = float(np.percentile(k_dist, p))
        db       = DBSCAN(eps=eps_cand, min_samples=min_samples).fit(coords)
        labels   = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters > 1:
            sil = silhouette_score(coords, labels)
            print(f"Percentil {p:2d}% → eps={eps_cand:.3f}, clusters={n_clusters}, silhouette={sil:.3f}")
            if sil > best["sil"]:
                best.update(eps=eps_cand, sil=sil, labels=labels.copy())
        else:
            print(f"Percentil {p:2d}% → eps={eps_cand:.3f}, clusters={n_clusters}")

    if best["eps"] is None:
        best["eps"]    = float(np.percentile(k_dist, 50))
        db             = DBSCAN(eps=best["eps"], min_samples=min_samples).fit(coords)
        best["labels"] = db.labels_
        print(f"⚠️ Ningún eps produjo >1 cluster; usando 50% = {best['eps']:.3f}")
    else:
        print(f"✅ Mejor eps = {best['eps']:.3f}, sil = {best['sil']:.3f}")

    labels     = best["labels"]
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = list(labels).count(-1)
    print(f"→ key_area_{i}: clusters = {n_clusters}, ruido = {n_noise}")

    df = pd.DataFrame({
        'id':      ids,
        'type':    types,
        'x':       coords[:, 0],
        'y':       coords[:, 1],
        'z':       coords[:, 2],
        'cluster': labels
    })
    csv_out = os.path.join(csv_dir, f"key_area_{i}_clustered.csv")
    df.to_csv(csv_out, index=False)
    print(f"✅ CSV con cluster actualizado: {csv_out}")

    def _set_cluster(frame, data):
        data.particles_.create_property('Cluster', data=labels.astype(np.int32))
        yield 'Asignando nuevas etiquetas de cluster'
    pipeline2 = import_file(file_path)
    pipeline2.modifiers.append(PythonScriptModifier(function=_set_cluster))
    out_dump = os.path.join(input_dir, f"key_area_{i}_clustered.dump")
    export_file(
        pipeline2,
        out_dump,
        "lammps/dump",
        columns=[
            "Particle Identifier",
            "Particle Type",
            "Position.X",
            "Position.Y",
            "Position.Z",
            "Cluster"
        ]
    )
    print(f"✅ Dump con cluster actualizado: {out_dump}")

    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection='3d')
    unique_labels = set(labels)
    palette       = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, palette):
        if k == -1:
            col = [0, 0, 0, 1]
        mask = (labels == k)
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            coords[mask, 2],
            c=[col],
            s=20,
            label=f"Cluster {k}"
        )
    ax.set_title(f"DBSCAN 3D key_area_{i} (eps={best['eps']:.3f}): {n_clusters} clusters")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.legend(loc="upper right", fontsize="small")
    plt.show()

