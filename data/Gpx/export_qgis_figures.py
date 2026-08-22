"""Create and export the frozen Quantum Spectral Ecology figure layouts.

Run from the QGIS Python console:

    from pathlib import Path
    PROJECT_ROOT = Path("/absolute/path/to/quantum-spectral-ecology").resolve()
    exec((PROJECT_ROOT / "data/Gpx/export_qgis_figures.py").read_text())

The script reads the styled layers already stored in the final GeoPackage.  It
creates a clean QGIS project containing five persistent print layouts, exports
each layout as PDF and 300 dpi PNG, and writes a checksum manifest.  It does
not modify the source GeoPackage.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from qgis.core import (
    Qgis,
    QgsFillSymbol,
    QgsLayoutExporter,
    QgsLayoutItemLabel,
    QgsLayoutItemLegend,
    QgsLayoutItemMap,
    QgsLayoutItemScaleBar,
    QgsLayoutPoint,
    QgsLayoutSize,
    QgsMapLayer,
    QgsPrintLayout,
    QgsProject,
    QgsRasterLayer,
    QgsRectangle,
    QgsSingleSymbolRenderer,
    QgsUnitTypes,
    QgsVectorLayer,
)
from qgis.PyQt.QtGui import QFont

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

PROJECT_ROOT = (
    Path(globals()["PROJECT_ROOT"])
    if "PROJECT_ROOT" in globals()
    else Path(__file__).resolve().parents[2]
)
SOURCE_GPKG = PROJECT_ROOT / "data/Gpx/GL_test2_radical1_results.gpkg"
OUTPUT_PROJECT = PROJECT_ROOT / "data/Gpx/GL_test2_radical1_figures.qgz"
OUTPUT_DIR = PROJECT_ROOT / "figures/generated"

GOOGLE_SATELLITE_URI = (
    "crs=EPSG:3857&format&type=xyz&"
    "url=https://mt1.google.com/vt/lyrs%3Ds%26x%3D%7Bx%7D%26y%3D%7By%7D%26z%3D%7Bz%7D&"
    "zmax=20&zmin=0"
)

EXPORT_DPI = 300
EXTENT_PADDING = 1.04
SCALE_SEGMENT_METERS = 100

PAGE_WIDTH_MM = 297.0
PAGE_HEIGHT_MM = 210.0
MAP_X_MM = 8.0
MAP_Y_MM = 15.0
MAP_WIDTH_MM = 225.0
MAP_HEIGHT_MM = 187.0
LEGEND_X_MM = 239.0
LEGEND_Y_MM = 18.0
LEGEND_WIDTH_MM = 50.0
LEGEND_HEIGHT_MM = 166.0


class FigureSpec:
    """Small immutable-like container which also works under QGIS ``exec``."""

    __slots__ = (
        "context_layers",
        "layout_name",
        "primary_layer",
        "show_legend",
        "stem",
        "title",
    )

    def __init__(
        self,
        *,
        stem: str,
        layout_name: str,
        title: str,
        primary_layer: str,
        context_layers: tuple[str, ...],
        show_legend: bool = True,
    ) -> None:
        self.stem = stem
        self.layout_name = layout_name
        self.title = title
        self.primary_layer = primary_layer
        self.context_layers = context_layers
        self.show_legend = show_legend


FIGURES = (
    FigureSpec(
        stem="01_study_grid",
        layout_name="01 Study grid",
        title="Study grid",
        primary_layer="grid_input",
        context_layers=("river", "grid_delimiter"),
        show_legend=False,
    ),
    FigureSpec(
        stem="02_smoothed_observations",
        layout_name="02 Smoothed observations",
        title="Smoothed observations",
        primary_layer="smoothed_observations",
        context_layers=(
            "peuthysanota_observation",
            "river",
            "grid_delimiter",
        ),
    ),
    FigureSpec(
        stem="03_inverse_potential",
        layout_name="03 Inverse potential",
        title="Reconstructed inverse potential",
        primary_layer="inverse_potential",
        context_layers=(
            "peuthysanota_observation",
            "river",
            "grid_delimiter",
        ),
    ),
    FigureSpec(
        stem="04_driver_prediction",
        layout_name="04 Driver prediction",
        title="Driver prediction",
        primary_layer="driver_prediction",
        context_layers=("river", "grid_delimiter"),
    ),
    FigureSpec(
        stem="05_grey_prediction",
        layout_name="05 Grey prediction",
        title="Grey-corrected prediction",
        primary_layer="grey_prediction",
        context_layers=("river", "grid_delimiter"),
    ),
)

# These checks make it difficult to export a plausible-looking figure from the
# wrong package or from an incomplete intermediate run.
EXPECTED_FEATURES = {
    "B2GL": 151875,
    "G2GL": 151875,
    "R2GL": 151875,
    "grid_input": 2890,
    "smoothed_observations": 636,
    "inverse_potential": 636,
    "driver_prediction": 1942,
    "grey_prediction": 1942,
    "river": 5,
    "river_points": 2659,
    "peuthysanota_observation": 81,
}

DISPLAY_NAMES = {
    "B2GL": "Blue-channel samples",
    "G2GL": "Green-channel samples",
    "R2GL": "Red-channel samples",
    "grid_input": "Study grid",
    "smoothed_observations": "Smoothed observations",
    "inverse_potential": "Inverse potential",
    "driver_prediction": "Driver prediction",
    "grey_prediction": "Grey-corrected prediction",
    "river": "River",
    "river_points": "River sample points",
    "peuthysanota_observation": "Peuthysanota observations",
}


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_layers(project: QgsProject) -> dict[str, QgsMapLayer]:
    if not SOURCE_GPKG.is_file():
        fail(f"Source GeoPackage does not exist: {SOURCE_GPKG}")

    layers: dict[str, QgsMapLayer] = {}
    root = project.layerTreeRoot()
    result_group = root.addGroup("Model results")
    context_group = root.addGroup("Context")
    source_group = root.addGroup("Source measurements")

    result_names = {
        "smoothed_observations",
        "inverse_potential",
        "driver_prediction",
        "grey_prediction",
    }

    for source_name, expected_count in EXPECTED_FEATURES.items():
        layer = QgsVectorLayer(
            f"{SOURCE_GPKG}|layername={source_name}",
            DISPLAY_NAMES[source_name],
            "ogr",
        )
        if not layer.isValid():
            fail(f"Could not load {source_name!r} from {SOURCE_GPKG}")
        if layer.crs().authid() != "EPSG:31969":
            fail(f"Unexpected CRS for {source_name}: {layer.crs().authid()}")
        count = layer.featureCount()
        if count != expected_count:
            fail(
                f"Unexpected feature count for {source_name}: "
                f"expected {expected_count}, found {count}"
            )

        project.addMapLayer(layer, False)
        if source_name in result_names:
            group = result_group
        elif source_name in {"grid_input", "river", "peuthysanota_observation"}:
            group = context_group
        else:
            group = source_group
        node = group.addLayer(layer)
        node.setItemVisibilityChecked(
            source_name in {"grey_prediction", "river", "peuthysanota_observation"}
        )
        layers[source_name] = layer
        print(f"Verified {source_name}: {count} features")

    source_group.setItemVisibilityChecked(False)

    satellite = QgsRasterLayer(
        GOOGLE_SATELLITE_URI,
        "Google Satellite",
        "wms",
    )
    if not satellite.isValid():
        fail("Could not create the Google Satellite XYZ layer.")
    project.addMapLayer(satellite, False)
    context_group.addLayer(satellite).setItemVisibilityChecked(True)
    layers["google_satellite"] = satellite
    print("Verified Google Satellite XYZ layer")

    # Reopen the authoritative grid and give it an outline-only renderer.  It
    # sits above every result surface, keeping cell delimiters visible without
    # hiding the result colors or the satellite background.
    delimiter = QgsVectorLayer(
        f"{SOURCE_GPKG}|layername=grid_input",
        "Grid delimiter",
        "ogr",
    )
    if not delimiter.isValid():
        fail("Could not create the grid-delimiter overlay.")
    delimiter_symbol = QgsFillSymbol.createSimple(
        {
            "color": "0,0,0,0",
            "outline_color": "20,20,20,210",
            "outline_style": "solid",
            "outline_width": "0.10",
            "outline_width_unit": "MM",
        }
    )
    delimiter.setRenderer(QgsSingleSymbolRenderer(delimiter_symbol))
    project.addMapLayer(delimiter, False)
    context_group.addLayer(delimiter).setItemVisibilityChecked(True)
    layers["grid_delimiter"] = delimiter
    print("Created transparent grid-delimiter overlay")
    return layers


def common_extent(grid_layer: QgsVectorLayer) -> QgsRectangle:
    extent = QgsRectangle(grid_layer.extent())
    if extent.isEmpty() or extent.width() <= 0 or extent.height() <= 0:
        fail("The study-grid extent is empty.")
    extent.scale(EXTENT_PADDING)
    return extent


def add_title(layout: QgsPrintLayout, text: str) -> None:
    title = QgsLayoutItemLabel(layout)
    title.setId("title")
    title.setText(text)
    title.setFont(QFont("Noto Sans", 14))
    title.attemptMove(QgsLayoutPoint(MAP_X_MM, 3.0, QgsUnitTypes.LayoutMillimeters))
    title.attemptResize(
        QgsLayoutSize(MAP_WIDTH_MM, 9.0, QgsUnitTypes.LayoutMillimeters)
    )
    title.setFrameEnabled(False)
    layout.addLayoutItem(title)


def add_legend(
    layout: QgsPrintLayout,
    map_item: QgsLayoutItemMap,
    primary_layer: QgsMapLayer,
) -> None:
    legend = QgsLayoutItemLegend(layout)
    legend.setId("legend")
    legend.setTitle("")
    legend.setLinkedMap(map_item)
    legend.setAutoUpdateModel(False)
    legend_root = legend.model().rootGroup()
    legend_root.removeAllChildren()
    legend_root.addLayer(primary_layer)
    legend.setResizeToContents(False)
    legend.attemptMove(
        QgsLayoutPoint(LEGEND_X_MM, LEGEND_Y_MM, QgsUnitTypes.LayoutMillimeters)
    )
    legend.attemptResize(
        QgsLayoutSize(
            LEGEND_WIDTH_MM,
            LEGEND_HEIGHT_MM,
            QgsUnitTypes.LayoutMillimeters,
        )
    )
    legend.setFrameEnabled(False)
    layout.addLayoutItem(legend)


def add_scale_bar(layout: QgsPrintLayout, map_item: QgsLayoutItemMap) -> None:
    scale_bar = QgsLayoutItemScaleBar(layout)
    scale_bar.setId("scale_bar")
    scale_bar.setStyle("Single Box")
    scale_bar.setLinkedMap(map_item)
    scale_bar.setUnits(QgsUnitTypes.DistanceMeters)
    scale_bar.setUnitsPerSegment(SCALE_SEGMENT_METERS)
    scale_bar.setNumberOfSegments(2)
    scale_bar.setNumberOfSegmentsLeft(0)
    scale_bar.setUnitLabel("m")
    scale_bar.applyDefaultSize()
    scale_bar.attemptMove(QgsLayoutPoint(15.0, 188.0, QgsUnitTypes.LayoutMillimeters))
    layout.addLayoutItem(scale_bar)


def add_crs_note(layout: QgsPrintLayout) -> None:
    note = QgsLayoutItemLabel(layout)
    note.setId("crs_note")
    note.setText("SIRGAS 2000 / UTM zone 15N (EPSG:31969)")
    note.setFont(QFont("Noto Sans", 7))
    note.attemptMove(QgsLayoutPoint(239.0, 190.0, QgsUnitTypes.LayoutMillimeters))
    note.attemptResize(QgsLayoutSize(50.0, 12.0, QgsUnitTypes.LayoutMillimeters))
    note.setFrameEnabled(False)
    layout.addLayoutItem(note)


def make_layout(
    project: QgsProject,
    layers: dict[str, QgsMapLayer],
    extent: QgsRectangle,
    spec: FigureSpec,
) -> QgsPrintLayout:
    layout = QgsPrintLayout(project)
    layout.initializeDefaults()
    layout.setName(spec.layout_name)
    layout.renderContext().setDpi(EXPORT_DPI)

    page = layout.pageCollection().page(0)
    page.setPageSize(
        QgsLayoutSize(
            PAGE_WIDTH_MM,
            PAGE_HEIGHT_MM,
            QgsUnitTypes.LayoutMillimeters,
        )
    )

    primary = layers[spec.primary_layer]
    # QgsLayoutItemMap expects the topmost layer first.  Context overlays are
    # therefore followed by the result surface and finally the satellite base.
    map_layers = [layers[name] for name in spec.context_layers]
    map_layers.extend((primary, layers["google_satellite"]))

    map_item = QgsLayoutItemMap(layout)
    map_item.setId("map")
    map_item.attemptMove(
        QgsLayoutPoint(MAP_X_MM, MAP_Y_MM, QgsUnitTypes.LayoutMillimeters)
    )
    map_item.attemptResize(
        QgsLayoutSize(MAP_WIDTH_MM, MAP_HEIGHT_MM, QgsUnitTypes.LayoutMillimeters)
    )
    map_item.setCrs(layers["grid_input"].crs())
    map_item.setLayers(map_layers)
    map_item.setKeepLayerSet(True)
    map_item.zoomToExtent(extent)
    map_item.setFrameEnabled(True)
    layout.addLayoutItem(map_item)

    add_title(layout, spec.title)
    if spec.show_legend:
        add_legend(layout, map_item, primary)
    add_scale_bar(layout, map_item)
    add_crs_note(layout)
    return layout


def check_export(result: int, target: Path) -> None:
    if result != QgsLayoutExporter.Success:
        fail(f"QGIS export failed with code {result}: {target}")


def export_layout(layout: QgsPrintLayout, stem: str) -> tuple[Path, Path]:
    png_path = OUTPUT_DIR / f"{stem}.png"
    pdf_path = OUTPUT_DIR / f"{stem}.pdf"

    exporter = QgsLayoutExporter(layout)
    image_settings = QgsLayoutExporter.ImageExportSettings()
    image_settings.dpi = EXPORT_DPI
    check_export(exporter.exportToImage(str(png_path), image_settings), png_path)

    pdf_settings = QgsLayoutExporter.PdfExportSettings()
    pdf_settings.dpi = EXPORT_DPI
    check_export(exporter.exportToPdf(str(pdf_path), pdf_settings), pdf_path)
    print(f"Exported {png_path}")
    print(f"Exported {pdf_path}")
    return png_path, pdf_path


def write_manifest(
    layers: dict[str, QgsMapLayer],
    extent: QgsRectangle,
    exported: list[Path],
) -> tuple[Path, Path]:
    manifest_path = OUTPUT_DIR / "figure_manifest.json"
    payload = {
        "qgis_version": Qgis.QGIS_VERSION,
        "source_geopackage": str(SOURCE_GPKG.relative_to(PROJECT_ROOT)),
        "source_geopackage_sha256": sha256(SOURCE_GPKG),
        "project": str(OUTPUT_PROJECT.relative_to(PROJECT_ROOT)),
        "crs": layers["grid_input"].crs().authid(),
        "extent": {
            "xmin": extent.xMinimum(),
            "ymin": extent.yMinimum(),
            "xmax": extent.xMaximum(),
            "ymax": extent.yMaximum(),
        },
        "extent_padding_factor": EXTENT_PADDING,
        "png_dpi": EXPORT_DPI,
        "basemap": {
            "name": "Google Satellite",
            "provider": "wms/xyz",
            "uri": GOOGLE_SATELLITE_URI,
            "external_live_service": True,
        },
        "figures": [
            {
                "stem": spec.stem,
                "layout_name": spec.layout_name,
                "title": spec.title,
                "primary_layer": spec.primary_layer,
                "context_layers": list(spec.context_layers),
            }
            for spec in FIGURES
        ],
        "exports": {
            str(path.relative_to(PROJECT_ROOT)): sha256(path) for path in exported
        },
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {manifest_path}")

    checksum_path = OUTPUT_DIR / "SHA256SUMS"
    checksum_targets = [SOURCE_GPKG, OUTPUT_PROJECT, *exported, manifest_path]
    checksum_path.write_text(
        "".join(
            f"{sha256(path)}  {path.relative_to(PROJECT_ROOT)}\n"
            for path in checksum_targets
        ),
        encoding="utf-8",
    )
    print(f"Wrote {checksum_path}")
    return manifest_path, checksum_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PROJECT.parent.mkdir(parents=True, exist_ok=True)

    # A separate project keeps the export reproducible and does not alter the
    # user's currently open QGIS project.
    project = QgsProject()
    project.setFileName(str(OUTPUT_PROJECT))
    project.setFilePathStorage(Qgis.FilePathType.Relative)

    layers = load_layers(project)
    project.setCrs(layers["grid_input"].crs())
    extent = common_extent(layers["grid_input"])

    exported: list[Path] = []
    for spec in FIGURES:
        layout = make_layout(project, layers, extent, spec)
        if not project.layoutManager().addLayout(layout):
            fail(f"Could not add layout to project: {spec.layout_name}")
        exported.extend(export_layout(layout, spec.stem))

    if not project.write():
        fail(f"Could not save QGIS project: {OUTPUT_PROJECT}: {project.error()}")
    print(f"Saved project with persistent layouts: {OUTPUT_PROJECT}")

    _manifest, checksums = write_manifest(layers, extent, exported)
    print("Figure pipeline completed successfully.")
    print(f"From the repository root, verify with: sha256sum -c {checksums}")


main()
