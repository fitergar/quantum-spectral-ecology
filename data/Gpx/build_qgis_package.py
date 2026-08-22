"""Build the styled Quantum Spectral Ecology GeoPackage in QGIS.

Run this file from the QGIS Python console, for example:

    from pathlib import Path
    PROJECT_ROOT = Path("/absolute/path/to/quantum-spectral-ecology").resolve()
    exec((PROJECT_ROOT / "data/Gpx/build_qgis_package.py").read_text())

Edit only the paths in the CONFIGURATION section before running it.  The
script joins model outputs to the authoritative polygons by grid ``id``.
The inverse potential is joined by the exact (X, Y) cell-centre coordinates
because the current training-predictions CSV does not contain ``id``.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import processing
from qgis.core import (
    QgsFeature,
    QgsField,
    QgsGraduatedSymbolRenderer,
    QgsProject,
    QgsVectorLayer,
    QgsWkbTypes,
)
from qgis.PyQt.QtCore import QVariant

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

PROJECT_ROOT = (
    Path(globals()["PROJECT_ROOT"])
    if "PROJECT_ROOT" in globals()
    else Path(__file__).resolve().parents[2]
)

BASE_GPKG = PROJECT_ROOT / "data/Gpx/GL_base_complete.gpkg"
BASE_LAYER = "GL_base_grid"

AUXILIARY_LAYER_COUNTS = {
    "B2GL": 151875,
    "G2GL": 151875,
    "R2GL": 151875,
    "inegi_contours": 30,
    "peuthysanota_observation": 81,
    "river": 5,
    "river_points": 2659,
}

PREPARED_CSV = PROJECT_ROOT / "outputs/la_gloria/prepared.csv"
POTENTIAL_CSV = PROJECT_ROOT / "outputs/la_gloria/train_driver_predictions.csv"
DRIVER_CSV = PROJECT_ROOT / "outputs/la_gloria/pred_driver_localavg.csv"
GREY_CSV = PROJECT_ROOT / "outputs/la_gloria/pred_grey_localavg.csv"

PREPARED_STYLE = PROJECT_ROOT / "data/Gpx/styles/prepared_scale.qml"
PREDICTION_STYLE = PROJECT_ROOT / "data/Gpx/styles/predict_style.qml"
POTENTIAL_STYLE = PROJECT_ROOT / "data/Gpx/styles/pot_style.qml"

OUTPUT_GPKG = PROJECT_ROOT / "data/Gpx/GL_la_gloria_results.gpkg"
POTENTIAL_CLASSES = 16

# Refuse accidental deletion by default. Change to True only when deliberately
# regenerating OUTPUT_GPKG. BASE_GPKG is never overwritten.
ALLOW_OVERWRITE = False


def fail(message: str) -> None:
    raise RuntimeError(message)


def require_file(path: Path) -> None:
    if not path.is_file():
        fail(f"Required file does not exist: {path}")


def integer(value: str, *, label: str) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer for {label}: {value!r}") from exc


def number(value: str, *, label: str) -> float | None:
    if value is None or not str(value).strip():
        return None
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid number for {label}: {value!r}") from exc
    return result if math.isfinite(result) else None


def read_value_by_id(path: Path, value_field: str) -> dict[int, float | None]:
    values: dict[int, float | None] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"id", value_field}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            fail(f"{path.name} lacks columns: {sorted(missing)}")

        for row in reader:
            key = integer(row["id"], label=f"{path.name}:id")
            if key in values:
                fail(f"Duplicate id {key} in {path}")
            values[key] = number(row[value_field], label=f"{path.name}:{value_field}")
    return values


def read_value_by_xy(
    path: Path, value_field: str
) -> dict[tuple[int, int], float | None]:
    values: dict[tuple[int, int], float | None] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"X", "Y", value_field}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            fail(f"{path.name} lacks columns: {sorted(missing)}")

        for row in reader:
            key = (
                integer(row["X"], label=f"{path.name}:X"),
                integer(row["Y"], label=f"{path.name}:Y"),
            )
            if key in values:
                fail(f"Duplicate coordinates {key} in {path}")
            values[key] = number(row[value_field], label=f"{path.name}:{value_field}")
    return values


def make_result_layer(
    base: QgsVectorLayer,
    *,
    name: str,
    value_field: str,
    values: dict,
    key_fields: tuple[str, ...],
) -> QgsVectorLayer:
    geometry_name = QgsWkbTypes.displayString(base.wkbType())
    layer = QgsVectorLayer(f"{geometry_name}?crs={base.crs().authid()}", name, "memory")
    if not layer.isValid():
        fail(f"Could not create temporary layer: {name}")

    provider = layer.dataProvider()
    provider.addAttributes(
        [
            QgsField("id", QVariant.Int),
            QgsField("X", QVariant.Int),
            QgsField("Y", QVariant.Int),
            QgsField(value_field, QVariant.Double),
        ]
    )
    layer.updateFields()

    output_features: list[QgsFeature] = []
    for source in base.getFeatures():
        if key_fields == ("id",):
            key = int(source["id"])
        elif key_fields == ("X", "Y"):
            key = (int(source["X"]), int(source["Y"]))
        else:
            fail(f"Unsupported join key: {key_fields}")

        value = values.get(key)
        if value is None:
            continue

        feature = QgsFeature(layer.fields())
        feature.setGeometry(source.geometry())
        feature.setAttributes(
            [int(source["id"]), int(source["X"]), int(source["Y"]), value]
        )
        output_features.append(feature)

    provider.addFeatures(output_features)
    layer.updateExtents()
    if not output_features:
        fail(f"Join produced no features for {name}")
    print(f"{name}: {len(output_features)} polygons")
    return layer


def load_style(layer: QgsVectorLayer, path: Path) -> None:
    result = layer.loadNamedStyle(str(path))
    if isinstance(result, tuple) and len(result) >= 2 and not bool(result[1]):
        fail(f"Could not apply {path.name} to {layer.name()}: {result[0]}")
    layer.triggerRepaint()


def main() -> None:
    required_files = [
        BASE_GPKG,
        PREPARED_CSV,
        POTENTIAL_CSV,
        DRIVER_CSV,
        GREY_CSV,
        PREPARED_STYLE,
        PREDICTION_STYLE,
        POTENTIAL_STYLE,
    ]
    for path in required_files:
        require_file(path)

    if OUTPUT_GPKG.resolve() == BASE_GPKG.resolve():
        fail("OUTPUT_GPKG must differ from BASE_GPKG.")
    if OUTPUT_GPKG.exists() and not ALLOW_OVERWRITE:
        fail(
            f"Output already exists: {OUTPUT_GPKG}\n"
            "Choose another OUTPUT_GPKG or deliberately set ALLOW_OVERWRITE=True."
        )
    OUTPUT_GPKG.parent.mkdir(parents=True, exist_ok=True)

    base = QgsVectorLayer(f"{BASE_GPKG}|layername={BASE_LAYER}", "grid_input", "ogr")
    if not base.isValid():
        fail(f"Could not load {BASE_LAYER!r} from {BASE_GPKG}")
    if base.crs().authid() != "EPSG:31969":
        fail(f"Unexpected base-grid CRS: {base.crs().authid()}")
    if base.featureCount() != 2890:
        fail(f"Expected 2,890 base cells, found {base.featureCount()}")
    print("grid_input: 2890 verified polygons, EPSG:31969")

    auxiliary_layers = []

    for layer_name, expected_count in AUXILIARY_LAYER_COUNTS.items():
        layer = QgsVectorLayer(
            f"{BASE_GPKG}|layername={layer_name}",
            layer_name,
            "ogr",
        )
        if not layer.isValid():
            fail(f"Could not load auxiliary layer {layer_name!r}")
        if layer.crs().authid() != "EPSG:31969":
            fail(f"Unexpected CRS for {layer_name!r}: {layer.crs().authid()}")
        if layer.featureCount() != expected_count:
            fail(
                f"Expected {expected_count} features in {layer_name!r}, "
                f"found {layer.featureCount()}"
            )

        auxiliary_layers.append(layer)
        print(f"{layer_name}: {expected_count} verified features")

    prepared = make_result_layer(
        base,
        name="smoothed_observations",
        value_field="NUMPOINTS_SMOOTH",
        values=read_value_by_id(PREPARED_CSV, "NUMPOINTS_SMOOTH"),
        key_fields=("id",),
    )
    potential = make_result_layer(
        base,
        name="inverse_potential",
        value_field="V_target",
        values=read_value_by_xy(POTENTIAL_CSV, "V_target"),
        key_fields=("X", "Y"),
    )
    driver = make_result_layer(
        base,
        name="driver_prediction",
        value_field="Pop_est",
        values=read_value_by_id(DRIVER_CSV, "Pop_est"),
        key_fields=("id",),
    )
    grey = make_result_layer(
        base,
        name="grey_prediction",
        value_field="Pop_est",
        values=read_value_by_id(GREY_CSV, "Pop_est"),
        key_fields=("id",),
    )

    load_style(prepared, PREPARED_STYLE)
    load_style(driver, PREDICTION_STYLE)
    load_style(grey, PREDICTION_STYLE)

    load_style(potential, POTENTIAL_STYLE)
    renderer = potential.renderer()
    if not isinstance(renderer, QgsGraduatedSymbolRenderer):
        fail("Potential QML did not create a graduated renderer.")
    renderer.setClassAttribute("V_target")
    renderer.updateClasses(
        potential,
        QgsGraduatedSymbolRenderer.Quantile,
        POTENTIAL_CLASSES,
    )
    potential.triggerRepaint()

    result = processing.run(
        "native:package",
        {
            "LAYERS": [base, *auxiliary_layers, prepared, potential, driver, grey],
            "OVERWRITE": bool(OUTPUT_GPKG.exists()),
            "SAVE_STYLES": True,
            "SELECTED_FEATURES_ONLY": False,
            "EXPORT_RELATED_LAYERS": False,
            "OUTPUT": str(OUTPUT_GPKG),
        },
    )

    print(f"Created styled GeoPackage: {OUTPUT_GPKG}")
    for uri in result.get("OUTPUT_LAYERS", []):
        print(f"  {uri}")

    project = QgsProject.instance()
    group = project.layerTreeRoot().addGroup("QSE generated outputs")
    for layer_name in (
        "grid_input",
        "smoothed_observations",
        "inverse_potential",
        "driver_prediction",
        "grey_prediction",
    ):
        packaged = QgsVectorLayer(
            f"{OUTPUT_GPKG}|layername={layer_name}", layer_name, "ogr"
        )
        if not packaged.isValid():
            fail(f"Packaged layer could not be reopened: {layer_name}")
        project.addMapLayer(packaged, False)
        group.addLayer(packaged)

    print("Added packaged layers to the current QGIS project.")
    print("Review them, then save the QGIS project into the GeoPackage manually.")


main()
