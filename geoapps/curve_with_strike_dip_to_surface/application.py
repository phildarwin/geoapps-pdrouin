# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024-2025 Mira Geoscience Ltd.                                '
#                                                                              '
#  This file is part of geoapps.                                               '
#                                                                              '
#  geoapps is distributed under the terms and conditions of the MIT License    '
#  (see LICENSE file at the root of this source code package).                 '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from __future__ import annotations

import numpy as np
from time import time
from uuid import UUID

from geoh5py.ui_json.utils import monitored_directory_copy
from geoh5py.objects.points import Points
from geoh5py.objects.curve import Curve
from geoh5py.objects.surface import Surface
from geoh5py.objects.block_model import BlockModel

from geoapps import assets_path
from geoapps.base.selection import ObjectDataSelection
from geoapps.utils import warn_module_not_found


with warn_module_not_found():
    from ipywidgets.widgets import Button, HBox, Layout, Text, VBox, FloatText, Label, Output, HTML, Dropdown, SelectMultiple
    from scipy.spatial import Delaunay
    import trimesh


app_initializer = {
    "geoh5": r"C:\Users\pdrouin\OneDrive - MineSense Technologies Ltd\Desktop\code\00aa.geoh5",
    "objects": None,
    "data": None,
}


class CurveWithStrikeDipToSurface(ObjectDataSelection):
    """
    Application for creating closed 3D volumes from curves with strike/dip properties
    """

    def __init__(self, **kwargs):
        self.defaults.update(**app_initializer)
        self.defaults.update(**kwargs)

        # Point creation inputs
        self._point_name = Text("TestPoint", description="Name:")
        self._x_coord = FloatText(0.0, description="X:")
        self._y_coord = FloatText(0.0, description="Y:")
        self._z_coord = FloatText(0.0, description="Z:")

        # Curve selection - multiple curves
        self._available_curves = SelectMultiple(description="Select Curves:")
        self._refresh_curves = Button(description="Refresh Curves", button_style='info')

        # Block model selection
        self._available_block_models = SelectMultiple(description="Select Block Model:")
        self._refresh_block_models = Button(description="Refresh Block Models", button_style='info')
        self._update_block_model = Button(description="Update Block Model", button_style='primary')

        # Property selection (shown after curve selection)
        self._litho_property = Dropdown(description="Litho Property:")
        self._order_property = Dropdown(description="Order Property:")
        self._strike_property = Dropdown(description="Strike Property:")
        self._dip_property = Dropdown(description="Dip Property:")

        # Surface generation parameters
        self._projection_distance = FloatText(500.0, description="Projection Distance (m):", min=0.1, max=10000.0)

        # Action buttons
        self._process_curve = Button(description="Process Curve", button_style='success')
        self._export_surface = Button(description="Export to Surface", button_style='primary')

        # Point creation buttons
        self._create_point = Button(description="Create Point", button_style='success')
        self._test_live_link = Button(description="Test Live Link", button_style='info')
        self._clear_output = Button(description="Clear Messages", button_style='warning')

        # Output area for messages
        self._output_area = Output()

        # Initialize projected surface data - now for multiple surfaces
        self._projected_surfaces = []  # List of (vertices, cells, curve_name, litho_data, order_data) tuples for each curve

        # Connect button events
        self._create_point.on_click(self.create_point_click)
        self._test_live_link.on_click(self.test_live_link_click)
        self._clear_output.on_click(self.clear_output_click)
        self._refresh_curves.on_click(self.refresh_curves_click)
        self._refresh_block_models.on_click(self.refresh_block_models_click)
        self._update_block_model.on_click(self.update_block_model_click)
        self._process_curve.on_click(self.process_curve_click)
        self._export_surface.on_click(self.export_surface_click)

        super().__init__(**self.defaults)

        self.trigger.on_click(self.trigger_click)
        self.output_panel = VBox([self.trigger, self.live_link_panel])

        # Initialize curve list
        self.refresh_curves_click(None)

        # Initialize block model list
        self.refresh_block_models_click(None)

    @property
    def main(self):
        if self._main is None:
            self._main = VBox(
                [
                    self.project_panel,
                    HTML("<h4>Curve Selection:</h4>"),
                    HBox([self._refresh_curves]),
                    self._available_curves,
                    HTML("<h4>Block Model Selection:</h4>"),
                    HBox([self._refresh_block_models]),
                    self._available_block_models,
                    HTML("<h4>Surface Parameters:</h4>"),
                    self._projection_distance,
                    HBox([self._process_curve, self._export_surface, self._update_block_model]),
                    HTML("<h4>Point Creation:</h4>"),
                    HBox([self._point_name]),
                    HBox([self._x_coord, self._y_coord, self._z_coord]),
                    HBox([self._create_point, self._test_live_link, self._clear_output]),
                    HTML("<h4>Messages:</h4>"),
                    self._output_area,
                    HTML("<h4>Actions:</h4>"),
                    self.output_panel,
                ]
            )
        return self._main

    @property
    def point_name(self):
        """Name for the point object"""
        return self._point_name

    @property
    def x_coord(self):
        """X coordinate input"""
        return self._x_coord

    @property
    def y_coord(self):
        """Y coordinate input"""
        return self._y_coord

    @property
    def z_coord(self):
        """Z coordinate input"""
        return self._z_coord

    def log_message(self, message, msg_type="info"):
        """Log a message to the output area"""
        with self._output_area:
            if msg_type == "success":
                print(f"✅ {message}")
            elif msg_type == "error":
                print(f"❌ {message}")
            elif msg_type == "warning":
                print(f"⚠️ {message}")
            else:
                print(f"ℹ️ {message}")

    def create_point_click(self, _):
        """Create a single point at specified coordinates"""
        try:
            # Get coordinates
            x = self.x_coord.value
            y = self.y_coord.value
            z = self.z_coord.value
            name = self.point_name.value or "TestPoint"

            # Create point data
            vertices = np.array([[x, y, z]])

            # Check live link status before creating workspace
            live_link_active = self.live_link.value
            self.log_message(f"Live link status: {'Active' if live_link_active else 'Inactive'}", "info")

            # Create output workspace
            temp_geoh5 = f"{name}_{time():.0f}.geoh5"
            export_path = self.export_directory.selected_path or "."
            ws, live_link_status = self.get_output_workspace(
                live_link_active, export_path, temp_geoh5
            )

            with ws as workspace:
                # Create points object
                points = Points.create(
                    workspace,
                    name=name,
                    vertices=vertices
                )

                self.log_message(f"Created point '{name}' at ({x}, {y}, {z})", "success")

                if live_link_status:
                    self.log_message("Live link active - point sent to Geoscience ANALYST", "success")
                    monitored_directory_copy(export_path, points)
                else:
                    self.log_message("Live link inactive - point saved to file only", "warning")

        except Exception as e:
            self.log_message(f"Error creating point: {e}", "error")
            import traceback
            self.log_message(f"Traceback: {traceback.format_exc()}", "error")

    def refresh_curves_click(self, _):
        """Refresh the list of available curves from the workspace"""
        try:
            if self.workspace is None:
                self.log_message("No workspace loaded. Please select a project first.", "warning")
                return

            # Get all curve objects from the workspace
            curves = []
            for obj in self.workspace.objects:
                if isinstance(obj, Curve):
                    curves.append((f"{obj.name} ({obj.uid})", obj.uid))

            if curves:
                self._available_curves.options = curves
                self._available_curves.value = ()  # Empty tuple for SelectMultiple
                self.log_message(f"✅ Found {len(curves)} curves in workspace", "success")
                self.log_message("💡 Select one or more curves from the list above", "info")
            else:
                self._available_curves.options = []
                self._available_curves.value = ()
                self.log_message("⚠️ No curves found in workspace", "warning")
                self.log_message("💡 Make sure your .geoh5 file contains curve objects", "info")

        except Exception as e:
            self.log_message(f"❌ Error refreshing curves: {e}", "error")
            import traceback
            self.log_message(f"🔍 Details: {traceback.format_exc()}", "error")

    def refresh_block_models_click(self, _):
        """Refresh the list of available block models from the workspace"""
        try:
            if self.workspace is None:
                self.log_message("No workspace loaded. Please select a project first.", "warning")
                return

            # Get all block model objects from the workspace
            block_models = []
            for obj in self.workspace.objects:
                if isinstance(obj, BlockModel):
                    block_models.append((f"{obj.name} ({obj.uid})", obj.uid))

            if block_models:
                self._available_block_models.options = block_models
                self._available_block_models.value = ()  # Empty tuple for SelectMultiple
                self.log_message(f"✅ Found {len(block_models)} block models in workspace", "success")
                self.log_message("💡 Select a block model to update with litho properties", "info")
            else:
                self._available_block_models.options = []
                self._available_block_models.value = ()
                self.log_message("⚠️ No block models found in workspace", "warning")
                self.log_message("💡 Make sure your .geoh5 file contains block model objects", "info")

        except Exception as e:
            self.log_message(f"❌ Error refreshing block models: {e}", "error")
            import traceback
            self.log_message(f"🔍 Details: {traceback.format_exc()}", "error")

    def update_block_model_click(self, _):
        """Update block model with litho properties from surfaces based on order"""
        try:
            if not self._projected_surfaces:
                self.log_message("No projected surfaces available. Process curves first.", "warning")
                return

            if not self._available_block_models.value:
                self.log_message("No block model selected. Please select a block model.", "warning")
                return

            self.log_message("🔄 Updating block model with litho properties...", "info")

            if self.workspace is None:
                self.log_message("No workspace loaded. Please select a project first.", "warning")
                return

            # Get the selected block model
            block_model_uid = self._available_block_models.value[0]  # Take first selected
            block_model = self.workspace.get_entity(block_model_uid)[0]

            if block_model is None or not isinstance(block_model, BlockModel):
                self.log_message("❌ Invalid block model selected", "error")
                return

            # Sort surfaces by order (ascending - lower order first, higher order overwrites)
            sorted_surfaces = sorted(self._projected_surfaces,
                                   key=lambda x: x[4][0] if x[4] is not None else 0)

            # Initialize litho array for block model
            litho_values = np.full(block_model.n_cells, np.nan)  # Start with NaN

            total_cells_updated = 0

            # Process each surface in order
            for surface_idx, (vertices, cells, curve_name, litho_data, order_data) in enumerate(sorted_surfaces):
                if litho_data is None:
                    self.log_message(f"⚠️ Skipping surface {curve_name} - no litho data", "warning")
                    continue

                litho_value = litho_data[0]  # Use first litho value
                order_value = order_data[0] if order_data is not None else 0

                self.log_message(f"📊 Processing surface {curve_name} (order: {order_value}, litho: {litho_value})", "info")

                # Find block model cells that intersect with this surface
                intersecting_cells = self._find_intersecting_cells(block_model, vertices, cells)

                if len(intersecting_cells) > 0:
                    # Update litho values for intersecting cells (overwrites previous values)
                    litho_values[intersecting_cells] = litho_value
                    total_cells_updated += len(intersecting_cells)
                    self.log_message(f"✅ Updated {len(intersecting_cells)} cells with litho value {litho_value}", "success")
                else:
                    self.log_message(f"⚠️ No intersecting cells found for surface {curve_name}", "warning")

            # Create output workspace for the new block model
            temp_geoh5 = f"block_model_with_litho_{time():.0f}.geoh5"
            export_path = self.export_directory.selected_path or "."
            ws, live_link_status = self.get_output_workspace(
                self.live_link.value, export_path, temp_geoh5
            )

            # Create a new block model with the same structure but with litho data
            new_block_model_name = f"{block_model.name}_with_litho_{time():.0f}"

            with ws as workspace:
                # Create new block model with same structure
                new_block_model = BlockModel.create(
                    workspace,
                    name=new_block_model_name,
                    origin=block_model.origin,
                    u_cell_delimiters=block_model.u_cell_delimiters,
                    v_cell_delimiters=block_model.v_cell_delimiters,
                    z_cell_delimiters=block_model.z_cell_delimiters,
                )

                # Copy all existing data from the original block model
                if hasattr(block_model, 'get_data_list') and block_model.get_data_list():
                    for data_name in block_model.get_data_list():
                        try:
                            data_obj = block_model.get_data(data_name)[0]
                            new_block_model.add_data({
                                data_name: {"values": data_obj.values}
                            })
                            self.log_message(f"✅ Copied data '{data_name}' from original block model", "info")
                        except Exception as e:
                            self.log_message(f"⚠️ Could not copy data '{data_name}': {e}", "warning")

                # Add the litho data from surfaces
                if total_cells_updated > 0:
                    new_block_model.add_data({
                        "litho_from_surfaces": {"values": litho_values}
                    })
                    self.log_message(f"📊 Successfully created new block model '{new_block_model_name}' with {total_cells_updated} cells updated", "success")
                else:
                    self.log_message("⚠️ No cells were updated in the new block model", "warning")

                # Export via live link if active
                if live_link_status:
                    self.log_message("Live link active - exporting to Geoscience ANALYST", "success")
                    monitored_directory_copy(export_path, new_block_model)
                    self.log_message("✅ New block model exported", "info")
                else:
                    self.log_message("New block model saved to file", "info")

        except Exception as e:
            self.log_message(f"❌ Error updating block model: {e}", "error")
            import traceback
            self.log_message(f"🔍 Details: {traceback.format_exc()}", "error")

    def _find_intersecting_cells(self, block_model, vertices, cells):
        """Find block model cells that intersect with the given surface using trimesh"""
        try:
            # Check if trimesh is available
            if 'trimesh' not in globals():
                self.log_message("🔄 Trimesh not available, falling back to Delaunay method", "warning")
                return self._find_intersecting_cells_delaunay(block_model, vertices, cells)

            # Get block model cell centers
            cell_centers = block_model.centroids

            # Create trimesh object from surface
            # Convert triangle indices to the format trimesh expects
            faces = cells.astype(int)

            # Create the mesh
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

            # Check if mesh is watertight (closed surface)
            if not mesh.is_watertight:
                self.log_message(f"⚠️ Surface mesh is not watertight, results may be inaccurate", "warning")

            # Use trimesh to check which points are inside the mesh
            # trimesh.contains() returns boolean array for each point
            points_inside = mesh.contains(cell_centers)

            # Get indices of cells that are inside the surface
            intersecting_cells = np.where(points_inside)[0]

            self.log_message(f"✅ Found {len(intersecting_cells)} cells inside surface using trimesh", "info")

            return intersecting_cells

        except Exception as e:
            self.log_message(f"❌ Error in trimesh intersection: {e}", "error")
            # Fallback to the old method if trimesh fails
            self.log_message("🔄 Falling back to Delaunay method", "warning")
            return self._find_intersecting_cells_delaunay(block_model, vertices, cells)

    def _find_intersecting_cells_delaunay(self, block_model, vertices, cells):
        """Fallback method using Delaunay triangulation for intersection testing"""
        try:
            # Get block model cell centers
            cell_centers = block_model.centroids

            # Create a 2D projection of the surface for intersection testing
            # Use XY plane for horizontal surfaces (most common for geological surfaces)
            surface_points_2d = vertices[:, :2]  # Take only X, Y coordinates

            # Create Delaunay triangulation of the surface
            try:
                tri = Delaunay(surface_points_2d)
            except Exception as e:
                self.log_message(f"⚠️ Could not create Delaunay triangulation: {e}", "warning")
                # Fall back to bounding box approach
                return self._find_intersecting_cells_bbox(block_model, vertices, cells)

            intersecting_cells = []

            # Test each block model cell center
            for cell_idx, cell_center in enumerate(cell_centers):
                # Project cell center to 2D
                cell_center_2d = cell_center[:2]

                # Find which triangle this point belongs to
                triangle_idx = tri.find_simplex(cell_center_2d)

                if triangle_idx >= 0:  # Point is inside the triangulation
                    # Get the actual triangle vertices in 3D
                    triangle = cells[triangle_idx]
                    tri_vertices_3d = vertices[triangle]

                    # Check if the point is within the Z bounds of the triangle
                    z_min = np.min(tri_vertices_3d[:, 2])
                    z_max = np.max(tri_vertices_3d[:, 2])

                    if z_min <= cell_center[2] <= z_max:
                        # Perform 3D point-in-triangle test
                        if self._point_in_triangle_3d(cell_center, tri_vertices_3d):
                            intersecting_cells.append(cell_idx)

            return np.array(intersecting_cells)

        except Exception as e:
            self.log_message(f"❌ Error in Delaunay intersection: {e}", "error")
            # Fallback to bounding box method
            self.log_message("🔄 Falling back to bounding box method", "warning")
            return self._find_intersecting_cells_bbox(block_model, vertices, cells)

    def _find_intersecting_cells_bbox(self, block_model, vertices, cells):
        """Last resort fallback method using bounding box intersection"""
        try:
            cell_centers = block_model.centroids

            # For each triangle in the surface, check if cell centers are inside
            intersecting_cells = []

            for triangle in cells:
                # Get triangle vertices
                tri_vertices = vertices[triangle]

                # Use simple bounding box check
                tri_min = np.min(tri_vertices, axis=0)
                tri_max = np.max(tri_vertices, axis=0)

                # Find cells within bounding box
                in_bbox = np.all((cell_centers >= tri_min) & (cell_centers <= tri_max), axis=1)
                candidate_cells = np.where(in_bbox)[0]

                # For each candidate cell, check if center is inside triangle
                for cell_idx in candidate_cells:
                    if cell_idx not in intersecting_cells:
                        cell_center = cell_centers[cell_idx]
                        if self._point_in_triangle_3d(cell_center, tri_vertices):
                            intersecting_cells.append(cell_idx)

            return np.array(intersecting_cells)

        except Exception as e:
            self.log_message(f"❌ Error in bbox intersection: {e}", "error")
            return np.array([])

    def _point_in_triangle_3d(self, point, triangle_vertices):
        """Check if a 3D point is inside a 3D triangle using barycentric coordinates"""
        try:
            # Get triangle vertices
            v0, v1, v2 = triangle_vertices

            # Compute vectors
            v0v1 = v1 - v0
            v0v2 = v2 - v0
            v0p = point - v0

            # Compute dot products
            dot00 = np.dot(v0v2, v0v2)
            dot01 = np.dot(v0v2, v0v1)
            dot02 = np.dot(v0v2, v0p)
            dot11 = np.dot(v0v1, v0v1)
            dot12 = np.dot(v0v1, v0p)

            # Compute barycentric coordinates
            inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
            u = (dot11 * dot02 - dot01 * dot12) * inv_denom
            v = (dot00 * dot12 - dot01 * dot02) * inv_denom

            # Check if point is in triangle
            return (u >= 0) and (v >= 0) and (u + v <= 1)

        except Exception as e:
            self.log_message(f"❌ Error in point-in-triangle test: {e}", "error")
            return False

    def on_curve_selected(self, change):
        """Handle curve selection from dropdown and populate property options"""
        if change['new']:
            try:
                if self.workspace is None:
                    self.log_message("No workspace loaded", "warning")
                    return

                # Get the selected curve
                curve_uid = change['new']
                curve = self.workspace.get_entity(curve_uid)[0]

                if curve and isinstance(curve, Curve):
                    self.log_message(f"✅ Selected curve: {curve.name}", "success")

                    # Get data attributes from the curve
                    data_names = []
                    try:
                        # Get data associated with this curve from the workspace
                        for data_obj in self.workspace.data:
                            if hasattr(data_obj, 'parent') and data_obj.parent == curve:
                                data_names.append(data_obj.name)
                            elif hasattr(data_obj, 'association') and data_obj.association == curve:
                                data_names.append(data_obj.name)

                        # Alternative: check all data objects in workspace that might be associated
                        if not data_names:
                            for data_obj in self.workspace.data:
                                # Check if data object name contains curve name or vice versa
                                if (curve.name.lower() in data_obj.name.lower() or
                                    data_obj.name.lower() in curve.name.lower()):
                                    data_names.append(data_obj.name)

                    except Exception as data_error:
                        self.log_message(f"⚠️ Error accessing curve data: {data_error}", "warning")
                        # Fallback: try to list all data in workspace
                        try:
                            for data_obj in self.workspace.data:
                                data_names.append(data_obj.name)
                        except:
                            pass

                    # Remove duplicates and sort
                    data_names = sorted(list(set(data_names)))

                    # Populate property dropdowns with available data
                    if data_names:
                        self._litho_property.options = data_names
                        self._order_property.options = data_names
                        self._strike_property.options = data_names
                        self._dip_property.options = data_names

                        # Auto-select common property names if they exist
                        if any('litho' in name.lower() for name in data_names):
                            litho_match = next(name for name in data_names if 'litho' in name.lower())
                            self._litho_property.value = litho_match
                        if any('order' in name.lower() for name in data_names):
                            order_match = next(name for name in data_names if 'order' in name.lower())
                            self._order_property.value = order_match
                        if any('strike' in name.lower() for name in data_names):
                            strike_match = next(name for name in data_names if 'strike' in name.lower())
                            self._strike_property.value = strike_match
                        if any('dip' in name.lower() for name in data_names):
                            dip_match = next(name for name in data_names if 'dip' in name.lower())
                            self._dip_property.value = dip_match

                        self.log_message(f"📊 Found {len(data_names)} data properties: {', '.join(data_names[:5])}{'...' if len(data_names) > 5 else ''}", "info")
                        self.log_message("💡 Select properties and click 'Process Curve'", "info")
                    else:
                        # Clear property dropdowns if no data
                        self._litho_property.options = []
                        self._order_property.options = []
                        self._strike_property.options = []
                        self._dip_property.options = []
                        self.log_message("⚠️ No data properties found in curve", "warning")
                        self.log_message("🔍 Try refreshing curves or check if your curve has associated data", "info")

            except Exception as e:
                self.log_message(f"❌ Error selecting curve: {e}", "error")
                import traceback
                self.log_message(f"🔍 Details: {traceback.format_exc()}", "error")

    def process_curve_click(self, _):
        """Process the selected curves by projecting segments downward using strike/dip"""
        try:
            if self.workspace is None:
                self.log_message("No workspace available", "error")
                return

            if not self._available_curves.value:
                self.log_message("No curves selected", "warning")
                return

            selected_curves = self._available_curves.value
            self.log_message(f"🔄 Processing {len(selected_curves)} curve(s)...", "info")

            # Clear previous results
            self._projected_surfaces = []

            # Process each selected curve using the EXACT same method
            for curve_idx, curve_uid in enumerate(selected_curves):
                try:
                    self.log_message(f"📊 Processing curve {curve_idx + 1}/{len(selected_curves)}", "info")

                    # Get the selected curve
                    curve = self.workspace.get_entity(curve_uid)[0]

                    if curve is None or not isinstance(curve, Curve):
                        self.log_message(f"❌ Invalid curve selected for curve {curve_idx + 1}", "error")
                        continue

                    # Get curve vertices
                    if not hasattr(curve, 'vertices') or curve.vertices is None:
                        self.log_message(f"❌ Curve {curve_idx + 1} has no vertices", "error")
                        continue

                    vertices = curve.vertices
                    self.log_message(f"📊 Curve {curve_idx + 1} has {len(vertices)} vertices", "info")

                    # Get strike and dip data
                    strike_data = None
                    dip_data = None
                    litho_data = None
                    order_data = None

                    if hasattr(curve, 'get_data_list') and curve.get_data_list():
                        for data_name in curve.get_data_list():
                            if data_name.lower() == 'strike':
                                strike_data = curve.get_data(data_name)[0].values
                            elif data_name.lower() == 'dip':
                                dip_data = curve.get_data(data_name)[0].values
                            elif data_name.lower() == 'litho':
                                litho_data = curve.get_data(data_name)[0].values
                            elif data_name.lower() == 'order':
                                order_data = curve.get_data(data_name)[0].values

                    if strike_data is None or dip_data is None:
                        self.log_message(f"❌ Strike or dip data not found on curve {curve_idx + 1}", "error")
                        continue

                    # Log litho and order data availability
                    if litho_data is not None:
                        self.log_message(f"📊 Found litho data on curve {curve_idx + 1}", "info")
                    else:
                        self.log_message(f"⚠️ No litho data found on curve {curve_idx + 1}", "warning")

                    if order_data is not None:
                        self.log_message(f"📊 Found order data on curve {curve_idx + 1}", "info")
                    else:
                        self.log_message(f"⚠️ No order data found on curve {curve_idx + 1}", "warning")

                    # Project each segment downward and create surfaces
                    projected_points = []
                    surface_cells = []  # Store surface connectivity
                    projection_distance = self._projection_distance.value  # Use the UI value

                    # Project all vertices first
                    projected_vertices = []
                    for i, vertex in enumerate(vertices):
                        # Get strike/dip at this vertex
                        strike_deg = float(strike_data[i])
                        dip_deg = float(dip_data[i])

                        # Convert to radians
                        strike_rad = np.radians(strike_deg)
                        dip_rad = np.radians(dip_deg)

                        # Calculate projection direction (perpendicular to strike, downward)
                        perp_dir = np.array([-np.cos(strike_rad), np.sin(strike_rad), 0])

                        # Apply dip (tilt downward)
                        dip_factor = np.sin(dip_rad)
                        horizontal_factor = np.cos(dip_rad)

                        # Final projection direction
                        proj_dir = horizontal_factor * perp_dir + dip_factor * np.array([0, 0, -1])
                        proj_dir = proj_dir / np.linalg.norm(proj_dir)  # Normalize

                        # Project the point
                        proj_point = vertex + projection_distance * proj_dir
                        projected_vertices.append(proj_point)

                        self.log_message(f"📍 Curve {curve_idx + 1}, Point {i+1}: strike={strike_deg:.1f}°, dip={dip_deg:.1f}°", "info")

                    projected_vertices = np.array(projected_vertices)

                    # Create top surface (original curve)
                    if len(vertices) >= 3:
                        # Calculate center point for top surface
                        top_center = np.mean(vertices, axis=0)
                        all_vertices = np.vstack([vertices, projected_vertices, top_center])
                        top_center_idx = len(vertices) + len(projected_vertices)

                        # Create triangles from center to each edge
                        for i in range(len(vertices)):
                            surface_cells.append([i, (i + 1) % len(vertices), top_center_idx])

                        self.log_message(f"✅ Created top surface for curve {curve_idx + 1} with {len(vertices)} triangles", "info")

                    # Create bottom surface (projected curve)
                    if len(projected_vertices) >= 3:
                        # Calculate center point for bottom surface
                        bottom_center = np.mean(projected_vertices, axis=0)
                        all_vertices = np.vstack([all_vertices, bottom_center])
                        bottom_center_idx = len(all_vertices) - 1

                        # Create triangles from center to each edge
                        for i in range(len(projected_vertices)):
                            surface_cells.append([len(vertices) + i, len(vertices) + ((i + 1) % len(projected_vertices)), bottom_center_idx])

                        self.log_message(f"✅ Created bottom surface for curve {curve_idx + 1} with {len(projected_vertices)} triangles", "info")

                    # Create side surfaces between consecutive points (including last to first)
                    for i in range(len(vertices)):
                        # Get the four corners of each surface quad
                        top_left = i
                        top_right = (i + 1) % len(vertices)  # Wrap around to first point
                        bottom_left = i + len(vertices)
                        bottom_right = ((i + 1) % len(vertices)) + len(vertices)

                        # Create two triangles for each quad (2 points, 2 triangles per pair)
                        surface_cells.extend([
                            [top_left, top_right, bottom_right],      # First triangle
                            [top_left, bottom_right, bottom_left]     # Second triangle
                        ])

                    self.log_message(f"✅ Created {len(vertices)} side surfaces for curve {curve_idx + 1} with 2 triangles each", "info")

                    if all_vertices.size > 0 and surface_cells:
                        total_triangles = len(surface_cells)
                        self.log_message(f"📊 Created complete closed surface for curve {curve_idx + 1} with {len(all_vertices)} vertices and {total_triangles} triangles", "info")

                        # Store the surface data for this curve
                        self._projected_surfaces.append((all_vertices, np.array(surface_cells), curve.name, litho_data, order_data))
                        self.log_message(f"✅ Curve {curve_idx + 1} processing completed!", "success")
                    else:
                        self.log_message(f"❌ No surfaces were created for curve {curve_idx + 1}", "error")

                except Exception as e:
                    self.log_message(f"❌ Error processing curve {curve_idx + 1}: {e}", "error")
                    continue

            if self._projected_surfaces:
                self.log_message(f"✅ All curves processed! Created {len(self._projected_surfaces)} surface(s)", "success")
                self.log_message("💡 Click 'Export to Surface' to save the results", "info")
            else:
                self.log_message("❌ No surfaces were created from any curves", "error")

        except Exception as e:
            self.log_message(f"❌ Error in curve processing: {e}", "error")
            import traceback
            self.log_message(f"🔍 Details: {traceback.format_exc()}", "error")

    def export_surface_click(self, _):
        """Export projected surfaces to workspace as separate objects"""
        try:
            if not self._projected_surfaces:
                self.log_message("No projected surface data available. Process curves first.", "warning")
                return

            self.log_message("🔄 Exporting projected surfaces as separate objects...", "info")

            # Create output workspace
            temp_geoh5 = f"projected_surfaces_{time():.0f}.geoh5"
            export_path = self.export_directory.selected_path or "."
            ws, live_link_status = self.get_output_workspace(
                self.live_link.value, export_path, temp_geoh5
            )

            total_vertices = 0
            total_cells = 0
            created_surfaces = []

            with ws as workspace:
                # Create separate Surface objects for each curve
                for surface_idx, surface_data in enumerate(self._projected_surfaces):
                    vertices, cells, curve_name, litho_data, order_data = surface_data

                    if len(vertices) == 0 or len(cells) == 0:
                        self.log_message(f"⚠️ Skipping empty surface for curve {curve_name}", "warning")
                        continue

                    # Create a unique name for this surface
                    surface_name = f"{curve_name}_Projected_{time():.0f}"

                    # Create individual Surface object
                    surface_obj = Surface.create(
                        workspace,
                        name=surface_name,
                        vertices=vertices,
                        cells=cells
                    )

                    # Add litho and order data to the surface if available
                    if litho_data is not None:
                        # Create litho data array with same value for all vertices
                        litho_values = np.full(len(vertices), litho_data[0])  # Use first litho value for all vertices
                        surface_obj.add_data({
                            "litho": {"values": litho_values}
                        })
                        self.log_message(f"✅ Added litho data to surface '{surface_name}'", "info")

                    if order_data is not None:
                        # Create order data array with same value for all vertices
                        order_values = np.full(len(vertices), order_data[0])  # Use first order value for all vertices
                        surface_obj.add_data({
                            "order": {"values": order_values}
                        })
                        self.log_message(f"✅ Added order data to surface '{surface_name}'", "info")

                    created_surfaces.append(surface_name)
                    total_vertices += len(vertices)
                    total_cells += len(cells)

                    self.log_message(f"✅ Created surface '{surface_name}' with {len(vertices)} vertices and {len(cells)} triangles", "success")

                if created_surfaces:
                    self.log_message(f"📊 Successfully created {len(created_surfaces)} separate surface objects with {total_vertices} total vertices and {total_cells} total triangles", "success")

                    # Export via live link if active
                    if live_link_status:
                        self.log_message("Live link active - exporting to Geoscience ANALYST", "success")
                        # Export all surfaces
                        for surface_name in created_surfaces:
                            surface_obj = workspace.get_entity(surface_name)[0]
                            if surface_obj:
                                monitored_directory_copy(export_path, surface_obj)
                        self.log_message("✅ All surfaces exported", "info")
                    else:
                        self.log_message("All surfaces saved to file", "info")
                else:
                    self.log_message("❌ No valid surfaces were created", "error")

        except Exception as e:
            self.log_message(f"❌ Error exporting surfaces: {e}", "error")
            import traceback
            self.log_message(f"🔍 Details: {traceback.format_exc()}", "error")

    def test_live_link_click(self, _):
        """Test live link connection"""
        try:
            if self.live_link.value:
                self.log_message("✅ Live link is active and ready!", "success")
                self.log_message(f"📁 Monitoring directory: {self.export_directory.selected_path}", "info")
                self.log_message("💡 Now try creating a point - it will be sent to Geoscience ANALYST!", "info")
            else:
                self.log_message("⚠️ Live link is inactive.", "warning")
                self.log_message("🔧 Enable it by checking the 'Geoscience ANALYST Pro - Live link' checkbox above.", "info")
        except Exception as e:
            self.log_message(f"❌ Error testing live link: {e}", "error")

    def clear_output_click(self, _):
        """Clear the output area"""
        self._output_area.clear_output()
        self.log_message("Output cleared", "info")

    def trigger_click(self, _):
        """Main trigger action - create a point at origin for testing"""
        self.log_message("Creating test point at origin...", "info")
        self.x_coord.value = 0.0
        self.y_coord.value = 0.0
        self.z_coord.value = 0.0
        self.point_name.value = "OriginPoint"
        self.create_point_click(None)

    def _project_point_downward(self, point, strike_deg, dip_deg, distance):
        """
        Project a point downward along the strike/dip direction by the specified distance.

        Args:
            point: 3D point coordinates [x, y, z]
            strike_deg: Strike angle in degrees (0-360)
            dip_deg: Dip angle in degrees (0-90)
            distance: Projection distance

        Returns:
            Projected point coordinates [x, y, z]
        """
        import math

        # Convert angles to radians
        strike_rad = math.radians(strike_deg)
        dip_rad = math.radians(dip_deg)

        # Calculate direction vector for downward projection
        # Strike direction (horizontal component)
        strike_x = math.sin(strike_rad)
        strike_y = math.cos(strike_rad)

        # Dip direction (vertical component)
        # For downward projection, we use negative z direction
        dip_x = strike_x * math.sin(dip_rad)
        dip_y = strike_y * math.sin(dip_rad)
        dip_z = -math.cos(dip_rad)  # Negative for downward

        # Normalize the direction vector
        direction_length = math.sqrt(dip_x**2 + dip_y**2 + dip_z**2)
        if direction_length > 0:
            dip_x /= direction_length
            dip_y /= direction_length
            dip_z /= direction_length

        # Project the point
        projected_point = np.array([
            point[0] + dip_x * distance,
            point[1] + dip_y * distance,
            point[2] + dip_z * distance
        ])

        return projected_point
