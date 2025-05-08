import os
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from ovito.io import import_file

# Crear directorio para CSV de salida
CSV_DIR = 'out/csv'
os.makedirs(CSV_DIR, exist_ok=True)

class CSVRecluster:
    def __init__(
        self,
        input_csv: str,
        min_samples: int = 5,
        percentiles: np.ndarray = np.arange(10, 100, 5)
    ):
        self.input_csv   = input_csv
        self.output_csv  = input_csv  # mismo nombre de entrada
        self.min_samples = min_samples
        self.percentiles = percentiles
        self.df          = None

    def load(self):
        """Carga el CSV en un DataFrame y verifica columnas."""
        self.df = pd.read_csv(self.input_csv)
        required = {'x', 'y', 'z', 'cluster'}
        if not required.issubset(self.df.columns):
            raise ValueError(f"Faltan columnas en {self.input_csv}; se requieren {required}")

    def _choose_best_eps(self, coords: np.ndarray):
        n_pts = len(coords)
        if n_pts < self.min_samples:
            return 0.0, np.zeros(n_pts, dtype=int)
        nbrs = NearestNeighbors(n_neighbors=self.min_samples).fit(coords)
        dists, _ = nbrs.kneighbors(coords)
        k_dist   = np.sort(dists[:, -1])
        best = {'sil': -1.0, 'eps': None, 'labels': None}
        for p in self.percentiles:
            eps_cand = float(np.percentile(k_dist, p))
            db       = DBSCAN(eps=eps_cand, min_samples=self.min_samples).fit(coords)
            labels   = db.labels_
            n_clust  = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clust > 1:
                sil = silhouette_score(coords, labels)
                if sil > best['sil']:
                    best.update(sil=sil, eps=eps_cand, labels=labels.copy())
        if best['eps'] is None:
            eps_med = float(np.percentile(k_dist, 50))
            labels  = DBSCAN(eps=eps_med, min_samples=self.min_samples).fit(coords).labels_
            best.update(eps=eps_med, labels=labels, sil=-1.0)
        return best['eps'], best['labels']

    def run(self):
        """Realiza el reclustering y sobrescribe el CSV de entrada."""
        self.load()
        df = self.df.copy()
        new_labels = np.zeros(len(df), dtype=int)
        global_id  = 0
        for orig in sorted(df['cluster'].unique()):
            mask       = (df['cluster'] == orig)
            coords_grp = df.loc[mask, ['x','y','z']].to_numpy()
            _, labels  = self._choose_best_eps(coords_grp)
            for sub in np.unique(labels):
                sub_mask = (labels == sub)
                idxs     = df.index[mask].to_numpy()[sub_mask]
                new_labels[idxs] = global_id
                global_id += 1
        df['cluster'] = new_labels
        df.to_csv(self.output_csv, index=False)
        print(f"✅ CSV actualizado: {self.output_csv}")
        return df


def dump_to_csv(dump_path: str, csv_path: str):
    """
    Convierte un archivo LAMMPS dump a CSV con columnas:
    id, type, x, y, z, cluster
    """
    pipeline = import_file(dump_path)
    data     = pipeline.compute()
    ids      = data.particles['Particle Identifier'].array
    types    = data.particles['Particle Type'].array
    coords   = data.particles.positions.array
    clusters = data.particles['Cluster'].array
    df = pd.DataFrame({
        'id': ids,
        'type': types,
        'x': coords[:,0],
        'y': coords[:,1],
        'z': coords[:,2],
        'cluster': clusters
    })
    df.to_csv(csv_path, index=False)
    print(f"✅ Dump convertido a CSV: {csv_path}")


def csv_to_dump(
    csv_path: str,
    template_dump: str,
    output_dump: str
):
    """
    Crea un archivo LAMMPS dump a partir de un CSV que contiene
    id, type, x, y, z, cluster. Conserva la cabecera del dump original.
    """
    # Leer toda la cabecera del dump original
    header_lines = []
    with open(template_dump) as fin:
        for line in fin:
            header_lines.append(line)
            if line.startswith("ITEM: ATOMS"):
                break
    # Leer CSV con pandas
    df = pd.read_csv(csv_path)
    # Escribir nuevo dump
    with open(output_dump, 'w') as fout:
        # Cabecera
        for hl in header_lines:
            fout.write(hl)
        # Filas
        for _, row in df.iterrows():
            fout.write(f"{int(row['id'])} {int(row['type'])} {row['x']} {row['y']} {row['z']} {int(row['cluster'])}\n")
    print(f"✅ Dump generado: {output_dump}")


def iterative_recluster(
    dump_path: str,
    max_iter: int = 10,
    min_samples: int = 5,
    percentiles: np.ndarray = np.arange(10, 100, 5)
) -> str:
    """
    Realiza reclustering iterativo y exporta dump final.
    """
    # Paths
    base       = os.path.splitext(os.path.basename(dump_path))[0]
    csv_path   = os.path.join(CSV_DIR, f"{base}.csv")
    final_dump = os.path.join(os.path.dirname(dump_path), f"{base}_clustered.dump")
    # 1) Dump -> CSV inicial
    dump_to_csv(dump_path, csv_path)
    # 2) Iterar reclustering sobre CSV
    prev_count = -1
    for it in range(1, max_iter+1):
        print(f"\n--- Iteración {it} ---")
        recluster = CSVRecluster(
            input_csv=csv_path,
            min_samples=min_samples,
            percentiles=percentiles
        )
        df = recluster.run()
        count = df['cluster'].nunique()
        print(f"Clusters tras iteración {it}: {count}")
        if count == prev_count:
            print("🔒 Divisiones estabilizadas.")
            break
        prev_count = count
    # 3) CSV final -> Dump con cabecera
    csv_to_dump(csv_path, dump_path, final_dump)
    print("\n✅ Proceso completo finalizado.")
    return final_dump

if __name__ == '__main__':

    for i in range(1,10):
        final_dump = iterative_recluster(
            dump_path=f'out/dump/key_area_{i}.dump',
            max_iter=5,
            min_samples=5,
            percentiles=np.arange(10, 100, 5)
        )
        print("Dump final reclusterizado:", final_dump)
