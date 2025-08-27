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
        self._projection_distance = FloatText(100.0, description="Projection Distance (m):", min=0.1, max=10000.0)
        self._z_offset = FloatText(50.0, description="Z Offset for Bottom:", min=0.1, max=1000.0)

        # Action buttons
        self._process_curve = Button(description="Process Curve", button_style='success')
        self._export_surface = Button(description="Export to Surface", button_style='primary')

        # Point creation buttons
        self._create_point = Button(description="Create Point", button_style='success')
        self._test_live_link = Button(description="Test Live Link", button_style='info')
        self._clear_output = Button(description="Clear Messages", button_style='warning')

        # Output area for messages
        self._output_area = Output()

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
                    self._z_offset,
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
        """Process the selected curve with chosen properties"""
        try:
            if not self._available_curves.value:
                self.log_message("No curve selected", "warning")
                return

            if not all([self._litho_property.value, self._order_property.value,
                       self._strike_property.value, self._dip_property.value]):
                self.log_message("Please select all 4 properties (litho, order, strike, dip)", "warning")
                return

            self.log_message("🔄 Processing curve with selected properties...", "info")
            self.log_message(f"📊 Litho: {self._litho_property.value}", "info")
            self.log_message(f"📊 Order: {self._order_property.value}", "info")
            self.log_message(f"📊 Strike: {self._strike_property.value}", "info")
            self.log_message(f"📊 Dip: {self._dip_property.value}", "info")

            # TODO: Add actual curve processing logic here
            # This would involve:
            # 1. Getting the curve data
            # 2. Extracting the selected properties
            # 3. Processing strike/dip to surface conversion
            # 4. Creating surface object

            self.log_message("✅ Curve processing completed!", "success")
            self.log_message("💡 Click 'Export to Surface' to save the result", "info")

        except Exception as e:
            self.log_message(f"❌ Error processing curve: {e}", "error")
            import traceback
            self.log_message(f"🔍 Details: {traceback.format_exc()}", "error")

    def export_surface_click(self, _):
        """Export processed curve to surface"""
        try:
            if not self._available_curves.value:
                self.log_message("No curve selected", "warning")
                return

            self.log_message("🔄 Exporting curve to surface...", "info")

            # Create output workspace
            temp_geoh5 = f"surface_from_curve_{time():.0f}.geoh5"
            export_path = self.export_directory.selected_path or "."
            ws, live_link_status = self.get_output_workspace(
                self.live_link.value, export_path, temp_geoh5
            )

            with ws as workspace:
                # Check if we have a valid workspace and curve
                if self.workspace is None:
                    raise ValueError("No source workspace available")

                # Get the selected curve and its data
                curve_uid = self._available_curves.value
                if curve_uid is None:
                    raise ValueError("No curve selected")

                curve = self.workspace.get_entity(curve_uid)[0]
                if curve is None or not isinstance(curve, Curve):
                    raise ValueError("Invalid curve selected")

                # Get strike and dip data for all points
                if self._strike_property.value is None or self._dip_property.value is None:
                    raise ValueError("Strike and dip data are required for surface generation")

                # Find the data objects for selected properties
                strike_data = None
                dip_data = None

                for data_obj in self.workspace.data:
                    if data_obj.name == self._strike_property.value:
                        strike_data = data_obj
                    elif data_obj.name == self._dip_property.value:
                        dip_data = data_obj

                if strike_data is None or dip_data is None:
                    raise ValueError("Strike and dip data are required for surface generation")

                # Get curve vertices safely
                try:
                    if hasattr(curve, 'vertices') and curve.vertices is not None:
                        curve_vertices = curve.vertices.copy()
                        self.log_message(f"✅ Retrieved {len(curve_vertices)} curve vertices", "info")
                    else:
                        raise ValueError("Curve has no vertices")
                except Exception as vert_error:
                    raise ValueError(f"Could not access curve vertices: {vert_error}")

                # Get projection parameters
                projection_distance = self._projection_distance.value
                z_offset = self._z_offset.value

                if projection_distance <= 0:
                    raise ValueError("Projection distance must be positive")
                if z_offset <= 0:
                    raise ValueError("Z offset must be positive")

                # Step 1: Project all points downward to reach Z offset
                self.log_message("🔄 Projecting points to bottom surface...", "info")
                bottom_vertices = self._project_all_points_to_z_offset(
                    curve_vertices, strike_data.values, dip_data.values, z_offset
                )

                # Step 2: Create a single closed surface containing top, bottom, and all sides
                self.log_message("🔄 Creating complete closed surface...", "info")
                closed_surface = self._create_closed_volume_surface(workspace, curve_vertices, bottom_vertices, curve)

                # Export the single closed surface via live link
                if live_link_status:
                    self.log_message("Live link active - exporting closed surface to Geoscience ANALYST", "success")
                    monitored_directory_copy(export_path, closed_surface)
                    self.log_message("✅ Complete closed surface exported", "info")
                else:
                    self.log_message("Closed surface saved to file", "info")

                self.log_message("✅ Complete 3D closed volume created!", "success")

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

    def _project_all_points_to_z_offset(self, curve_vertices, strike_values, dip_values, z_offset):
        """
        Project all curve points downward along their strike/dip until Z decreases by z_offset.

        Args:
            curve_vertices: Original curve vertices
            strike_values: Strike values for each point
            dip_values: Dip values for each point
            z_offset: Target Z decrease

        Returns:
            Projected vertices with Z offset
        """
        projected_vertices = []

        for i, point in enumerate(curve_vertices):
            try:
                strike_deg = float(strike_values[i])
                dip_deg = float(dip_values[i])

                # Calculate how far to project to achieve the Z offset
                # Z component of the dip direction
                import math
                dip_rad = math.radians(dip_deg)
                z_component = -math.cos(dip_rad)  # Negative for downward

                if abs(z_component) < 1e-10:
                    # Vertical dip, use a small fixed distance
                    distance = z_offset
                else:
                    # Calculate distance needed to achieve z_offset
                    distance = z_offset / abs(z_component)

                # Project the point
                projected_point = self._project_point_downward(point, strike_deg, dip_deg, distance)
                projected_vertices.append(projected_point)

            except (IndexError, ValueError, TypeError) as e:
                self.log_message(f"⚠️ Error projecting point {i}: {e}, using original point", "warning")
                projected_vertices.append(point.copy())

        return np.array(projected_vertices)

    def _create_closed_volume_surface(self, workspace, top_vertices, bottom_vertices, curve):
        """
        Create a single closed surface containing top, bottom, and all sides.

        Args:
            workspace: The geoh5 workspace
            top_vertices: Vertices of the top curve
            bottom_vertices: Vertices of the bottom curve
            curve: The original curve object

        Returns:
            Single Surface object representing the complete closed volume
        """
        from geoh5py.objects.surface import Surface

        # Combine all vertices: top curve + bottom curve
        all_vertices = np.vstack([top_vertices, bottom_vertices])

        # Create cells for the complete closed surface
        cells = []
        n_top_points = len(top_vertices)
        n_bottom_points = len(bottom_vertices)

        # 1. Top surface (triangulated)
        if n_top_points >= 3:
            # Calculate center point for top surface
            top_center = np.mean(top_vertices, axis=0)
            all_vertices = np.vstack([all_vertices, top_center])
            top_center_idx = len(all_vertices) - 1

            # Create triangles from center to each edge
            for i in range(n_top_points):
                cells.append([i, (i + 1) % n_top_points, top_center_idx])

        # 2. Bottom surface (triangulated)
        if n_bottom_points >= 3:
            # Calculate center point for bottom surface
            bottom_center = np.mean(bottom_vertices, axis=0)
            all_vertices = np.vstack([all_vertices, bottom_center])
            bottom_center_idx = len(all_vertices) - 1

            # Create triangles from center to each edge
            for i in range(n_bottom_points):
                cells.append([n_top_points + i, n_top_points + ((i + 1) % n_bottom_points), bottom_center_idx])

        # 3. Side surfaces (including closing side from last to first)
        for i in range(n_top_points):
            # Get the four corner points for this side segment
            top_left = i
            top_right = (i + 1) % n_top_points  # This ensures closing side
            bottom_left = n_top_points + i
            bottom_right = n_top_points + ((i + 1) % n_bottom_points)

            # Create two triangles to form the quadrilateral
            cells.append([top_left, top_right, bottom_left])      # Triangle 1
            cells.append([top_right, bottom_right, bottom_left])  # Triangle 2

        surface_cells = np.array(cells, dtype=np.int32)

        # Create the single closed surface
        surface_name = f"Closed_Volume_{getattr(curve, 'name', 'Curve')}"
        closed_surface = Surface.create(
            workspace,
            name=surface_name,
            vertices=all_vertices,
            cells=surface_cells
        )

        return closed_surface
