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

from geoapps import assets_path
from geoapps.base.selection import ObjectDataSelection
from geoapps.utils import warn_module_not_found


with warn_module_not_found():
    from ipywidgets.widgets import Button, HBox, Layout, Text, VBox, FloatText, Label, Output, HTML, Dropdown, SelectMultiple


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

        # Curve selection
        self._available_curves = Dropdown(description="Select Curve:")
        self._refresh_curves = Button(description="Refresh Curves", button_style='info')

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

        # Initialize projected surface data
        self._projected_surface_vertices = None
        self._projected_surface_cells = None

        # Connect button events
        self._create_point.on_click(self.create_point_click)
        self._test_live_link.on_click(self.test_live_link_click)
        self._clear_output.on_click(self.clear_output_click)
        self._refresh_curves.on_click(self.refresh_curves_click)
        self._process_curve.on_click(self.process_curve_click)
        self._export_surface.on_click(self.export_surface_click)
        self._available_curves.observe(self.on_curve_selected, names='value')

        super().__init__(**self.defaults)

        self.trigger.on_click(self.trigger_click)
        self.output_panel = VBox([self.trigger, self.live_link_panel])

        # Initialize curve list
        self.refresh_curves_click(None)

    @property
    def main(self):
        if self._main is None:
            self._main = VBox(
                [
                    self.project_panel,
                    HTML("<h4>Curve Selection:</h4>"),
                    HBox([self._refresh_curves]),
                    self._available_curves,
                    HTML("<h4>Property Selection:</h4>"),
                    self._litho_property,
                    self._order_property,
                    self._strike_property,
                    self._dip_property,
                    HTML("<h4>Surface Parameters:</h4>"),
                    self._projection_distance,
                    HBox([self._process_curve, self._export_surface]),
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
                self._available_curves.value = None  # Don't auto-select
                self.log_message(f"✅ Found {len(curves)} curves in workspace", "success")
                self.log_message("💡 Select a curve from the dropdown above", "info")
            else:
                self._available_curves.options = []
                self._available_curves.value = None
                self.log_message("⚠️ No curves found in workspace", "warning")
                self.log_message("💡 Make sure your .geoh5 file contains curve objects", "info")

        except Exception as e:
            self.log_message(f"❌ Error refreshing curves: {e}", "error")
            import traceback
            self.log_message(f"🔍 Details: {traceback.format_exc()}", "error")

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
        """Process the selected curve by projecting segments downward using strike/dip"""
        try:
            if self.workspace is None:
                self.log_message("No workspace available", "error")
                return
                
            if not self._available_curves.value:
                self.log_message("No curve selected", "warning")
                return

            self.log_message("🔄 Processing curve segments...", "info")

            # Get the selected curve
            curve_uid = self._available_curves.value
            curve = self.workspace.get_entity(curve_uid)[0]

            # Get the selected curve
            curve_uid = self._available_curves.value
            curve = self.workspace.get_entity(curve_uid)[0]
            
            if curve is None or not isinstance(curve, Curve):
                self.log_message("Invalid curve selected", "error")
                return

            # Get curve vertices
            if not hasattr(curve, 'vertices') or curve.vertices is None:
                self.log_message("Curve has no vertices", "error")
                return
                
            vertices = curve.vertices
            self.log_message(f"📊 Curve has {len(vertices)} vertices", "info")

            # Get strike and dip data
            strike_data = None
            dip_data = None
            
            if hasattr(curve, 'get_data_list') and curve.get_data_list():
                for data_name in curve.get_data_list():
                    if data_name.lower() == 'strike':
                        strike_data = curve.get_data(data_name)[0].values
                    elif data_name.lower() == 'dip':
                        dip_data = curve.get_data(data_name)[0].values

            if strike_data is None or dip_data is None:
                self.log_message("❌ Strike or dip data not found on curve", "error")
                return

            # Project each segment downward and create surfaces
            projected_points = []
            surface_cells = []  # Store surface connectivity
            projection_distance = 500.0  # Default 500m projection
            
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
                strike_dir = np.array([np.sin(strike_rad), np.cos(strike_rad), 0])
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
                
                self.log_message(f"� Point {i+1}: strike={strike_deg:.1f}°, dip={dip_deg:.1f}°", "info")
            
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
                
                self.log_message(f"✅ Created top surface with {len(vertices)} triangles", "info")
            
            # Create bottom surface (projected curve)
            if len(projected_vertices) >= 3:
                # Calculate center point for bottom surface
                bottom_center = np.mean(projected_vertices, axis=0)
                all_vertices = np.vstack([all_vertices, bottom_center])
                bottom_center_idx = len(all_vertices) - 1
                
                # Create triangles from center to each edge
                for i in range(len(projected_vertices)):
                    surface_cells.append([len(vertices) + i, len(vertices) + ((i + 1) % len(projected_vertices)), bottom_center_idx])
                
                self.log_message(f"✅ Created bottom surface with {len(projected_vertices)} triangles", "info")
            
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
            
            self.log_message(f"✅ Created {len(vertices)} side surfaces with 2 triangles each", "info")
            
            if all_vertices.size > 0 and surface_cells:
                total_triangles = len(surface_cells)
                self.log_message(f"📊 Created complete closed surface with {len(all_vertices)} vertices and {total_triangles} triangles", "info")
                
                # Store the surface data for export
                self._projected_surface_vertices = all_vertices
                self._projected_surface_cells = np.array(surface_cells)
                self.log_message("✅ Curve processing completed!", "success")
                self.log_message("💡 Click 'Export to Surface' to save the result", "info")
            else:
                self.log_message("❌ No surfaces were created", "error")

        except Exception as e:
            self.log_message(f"❌ Error processing curve: {e}", "error")
            import traceback
            self.log_message(f"🔍 Details: {traceback.format_exc()}", "error")

    def export_surface_click(self, _):
        """Export projected surface to workspace"""
        try:
            if self._projected_surface_vertices is None or self._projected_surface_cells is None:
                self.log_message("No projected surface data available. Process a curve first.", "warning")
                return

            self.log_message("🔄 Exporting projected surface...", "info")

            # Create output workspace
            temp_geoh5 = f"projected_surface_{time():.0f}.geoh5"
            export_path = self.export_directory.selected_path or "."
            ws, live_link_status = self.get_output_workspace(
                self.live_link.value, export_path, temp_geoh5
            )

            with ws as workspace:
                # Create a Surface object with the projected surface
                surface_obj = Surface.create(
                    workspace,
                    name=f"Projected_Surface_{time():.0f}",
                    vertices=self._projected_surface_vertices,
                    cells=self._projected_surface_cells
                )
                
                self.log_message(f"✅ Created surface with {len(self._projected_surface_vertices)} vertices and {len(self._projected_surface_cells)} triangles", "success")
                
                # Export via live link if active
                if live_link_status:
                    self.log_message("Live link active - exporting to Geoscience ANALYST", "success")
                    monitored_directory_copy(export_path, surface_obj)
                    self.log_message("✅ Surface exported", "info")
                else:
                    self.log_message("Surface saved to file", "info")

        except Exception as e:
            self.log_message(f"❌ Error exporting surface: {e}", "error")
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
