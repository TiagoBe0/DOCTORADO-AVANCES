import os
import json
import math
import numpy as np
import pandas as pd
import xgboost as xgb
from ovito.io import import_file, export_file
from ovito.modifiers import (
    ExpressionSelectionModifier,
    DislocationAnalysisModifier,
    DeleteSelectedModifier,
    ClusterAnalysisModifier,
    ConstructSurfaceModifier,
    InvertSelectionModifier
)

# Parámetros
radius = 2
sm_level = 8
cuttof_var = 3

# Crear carpeta de salida
output_dir = "out/dump"
os.makedirs(output_dir, exist_ok=True)

# Pipeline principal
pipeline = import_file("inputs.dump/dump-finalCool.160000")
pipeline.modifiers.append(ConstructSurfaceModifier(
    radius=radius,
    select_surface_particles=True,
    smoothing_level=sm_level
))
pipeline.modifiers.append(InvertSelectionModifier())
pipeline.modifiers.append(DeleteSelectedModifier())
pipeline.modifiers.append(ClusterAnalysisModifier(
    cutoff=cuttof_var,
    unwrap_particles=True,
    sort_by_size=True
))

# Computar para obtener etiquetas de clúster
data = pipeline.compute()
clusters = data.particles['Cluster'].array

# Contar clústeres (asumiendo IDs 1..N)
clusters_total = int(clusters.max())
print(f"Número de clústeres encontrados: {clusters_total}")

# Exportar dump con todos los clústeres
key_areas_path = os.path.join(output_dir, "key_areas.dump")
try:
    export_file(
        pipeline,
        key_areas_path,
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
    print(f"✅ Exportado: {key_areas_path}")
except Exception as e:
    print("Error en export_training_dump:", e)

# Dividir por clúster y exportar cada uno
pipeline_2 = import_file(key_areas_path)
for i in range(1, clusters_total + 1):
    # Selecciona solo el clúster i
    pipeline_2.modifiers.clear()
    pipeline_2.modifiers.append(ExpressionSelectionModifier(expression=f"Cluster == {i}"))
    pipeline_2.modifiers.append(InvertSelectionModifier())
    pipeline_2.modifiers.append(DeleteSelectedModifier())

    out_path = os.path.join(output_dir, f"key_area_{i}.dump")
    try:
        export_file(
            pipeline_2,
            out_path,
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
        print(f"✅ Exportado: {out_path}")
    except Exception as e:
        print(f"Error exportando key_area_{i}.dump:", e)

