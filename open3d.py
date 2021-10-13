# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------

import ipywidgets
import traitlets
import IPython
import json
import functools
import open3d as o3d
# Note: the _AsyncEventLoop is started whenever this module is imported.
from open3d.visualization._async_event_loop import _async_event_loop

from open3d._build_config import _build_config
if not _build_config["BUILD_JUPYTER_EXTENSION"]:
    raise RuntimeError(
        "Open3D WebVisualizer Jupyter extension is not available. To use "
        "WebVisualizer, build Open3D with -DBUILD_JUPYTER_EXTENSION=ON.")


@ipywidgets.register
class WebVisualizer(ipywidgets.DOMWidget):
    """Open3D Web Visualizer based on WebRTC."""

    # Name of the widget view class in front-end.
    _view_name = traitlets.Unicode('WebVisualizerView').tag(sync=True)

    # Name of the widget model class in front-end.
    _model_name = traitlets.Unicode('WebVisualizerModel').tag(sync=True)

    # Name of the front-end module containing widget view.
    _view_module = traitlets.Unicode('open3d').tag(sync=True)

    # Name of the front-end module containing widget model.
    _model_module = traitlets.Unicode('open3d').tag(sync=True)

    # Version of the front-end module containing widget view.
    # @...@ is configured by cpp/pybind/make_python_package.cmake.
    _view_module_version = traitlets.Unicode(
        '~@PROJECT_VERSION_THREE_NUMBER@').tag(sync=True)
    # Version of the front-end module containing widget model.
    _model_module_version = traitlets.Unicode(
        '~@PROJECT_VERSION_THREE_NUMBER@').tag(sync=True)

    # Widget specific property. Widget properties are defined as traitlets. Any
    # property tagged with `sync=True` is automatically synced to the frontend
    # *any* time it changes in Python. It is synced back to Python from the
    # frontend *any* time the model is touched.
    window_uid = traitlets.Unicode("window_UNDEFINED",
                                   help="Window UID").tag(sync=True)

    # Two-way communication channels.
    pyjs_channel = traitlets.Unicode(
        "Empty pyjs_channel.",
        help="Python->JS message channel.").tag(sync=True)
    jspy_channel = traitlets.Unicode(
        "Empty jspy_channel.",
        help="JS->Python message channel.").tag(sync=True)

    def show(self):
        IPython.display.display(self)

    def _call_http_api(self, entry_point, query_string, data):
        return o3d.visualization.webrtc_server.call_http_api(
            entry_point, query_string, data)

    @traitlets.validate('window_uid')
    def _valid_window_uid(self, proposal):
        if proposal['value'][:7] != "window_":
            raise traitlets.TraitError('window_uid must be "window_xxx".')
        return proposal['value']

    @traitlets.observe('jspy_channel')
    def _on_jspy_channel(self, change):
        # self.result_map = {"0": "result0",
        #                    "1": "result1", ...};
        if not hasattr(self, "result_map"):
            self.result_map = dict()

        jspy_message = change["new"]
        try:
            jspy_requests = json.loads(jspy_message)

            for call_id, payload in jspy_requests.items():
                if "func" not in payload or payload["func"] != "call_http_api":
                    raise ValueError(f"Invalid jspy function: {jspy_requests}")
                if "args" not in payload or len(payload["args"]) != 3:
                    raise ValueError(
                        f"Invalid jspy function arguments: {jspy_requests}")

                # Check if already in result.
                if not call_id in self.result_map:
                    json_result = self._call_http_api(payload["args"][0],
                                                      payload["args"][1],
                                                      payload["args"][2])
                    self.result_map[call_id] = json_result
        except:
            print(
                f"jspy_message is not a function call, ignored: {jspy_message}")
        else:
            self.pyjs_channel = json.dumps(self.result_map)


def draw(geometry=None,
         title="Open3D",
         width=640,
         height=480,
         actions=None,
         lookat=None,
         eye=None,
         up=None,
         field_of_view=60.0,
         bg_color=(1.0, 1.0, 1.0, 1.0),
         bg_image=None,
         show_ui=None,
         point_size=None,
         animation_time_step=1.0,
         animation_duration=None,
         rpc_interface=False,
         on_init=None,
         on_animation_frame=None,
         on_animation_tick=None):
    """Draw in Jupyter Cell"""

    window_uid = _async_event_loop.run_sync(
        functools.partial(import open3d as o3
#Help on class PointCloud in module open3d.open3d_pybind.geometry:

class PointCloud(Geometry3D):
  PointCloud class. A point cloud consists of point coordinates, and optionally point colors and point normals.

  Method resolution order:
      PointCloud
      Geometry3D
      Geometry
      pybind11_builtins.pybind11_object
      builtins.object

  Methods defined here:

  def _add_(...)
      _add_(self: open3d.open3d_pybind.geometry.PointCloud, arg0: open3d.open3d_pybind.geometry.PointCloud) -> open3d.open3d_pybind.geometry.PointCloud

  def _copy_(...)
      _copy_(self: open3d.open3d_pybind.geometry.PointCloud) -> open3d.open3d_pybind.geometry.PointCloud

  _deepcopy_(...)
      _deepcopy_(self: open3d.open3d_pybind.geometry.PointCloud, arg0: dict) -> open3d.open3d_pybind.geometry.PointCloud

  _iadd_(...)
      _iadd_(self: open3d.open3d_pybind.geometry.PointCloud, arg0: open3d.open3d_pybind.geometry.PointCloud) -> open3d.open3d_pybind.geometry.PointCloud

  _init_(...)
      _init_(*args, **kwargs)
      Overloaded function.

      1. _init_(self: open3d.open3d_pybind.geometry.PointCloud) -> None

      Default constructor

      2.         _init_(self: open3d.open3d_pybind.geometry.PointCloud, arg0: open3d.open3d_pybind.geometry.PointCloud) -> None

      Copy constructor

      3. _init_(self: open3d.open3d_pybind.geometry.PointCloud, points: open3d.open3d_pybind.utility.Vector3dVector) -> None

      Create a PointCloud from points

  _repr_(...)
      _repr_(self: open3d.open3d_pybind.geometry.PointCloud) -> str

  cluster_dbscan(...)
      cluster_dbscan(self, eps, min_points, print_progress=False)

      Cluster PointCloud using the DBSCAN algorithm  Ester et al., 'A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise', 1996. Returns a list of point labels, -1 indicates noise according to the algorithm.

      Args:
          eps (float): Density parameter that is used to find neighbouring points.
          min_points (int): Minimum number of points to form a cluster.
          print_progress (bool, optional, default=False): If true the progress is visualized in the console.

      Returns:
          open3d.utility.IntVector

  compute_convex_hull(...)
      compute_convex_hull(self)

      Computes the convex hull of the point cloud.

      Returns:
          Tuple[open3d.geometry.TriangleMesh, List[int]]

  compute_mahalanobis_distance(...)
      compute_mahalanobis_distance(self)

      Function to compute the Mahalanobis distance for points in a point cloud. See: https://en.wikipedia.org/wiki/Mahalanobis_distance.

      Returns:
          open3d.utility.DoubleVector

  compute_mean_and_covariance(...)
      compute_mean_and_covariance(self)

      Function to compute the mean and covariance matrix of a point cloud.

      Returns:
          Tuple[numpy.ndarray[float64[3, 1]], numpy.ndarray[float64[3, 3]]]

  compute_nearest_neighbor_distance(...)
      compute_nearest_neighbor_distance(self)

      Function to compute the distance from a point to its nearest neighbor in the point cloud

      Returns:
          open3d.utility.DoubleVector

  compute_point_cloud_distance(...)
      compute_point_cloud_distance(self, target)

      For each point in the source point cloud, compute the distance to the target point cloud.

      Args:
          target (open3d.geometry.PointCloud): The target point cloud.

      Returns:
          open3d.utility.DoubleVector

  crop(...)
      crop(bounding_box, bounding_box)

      Function to crop input pointcloud into output pointcloud

      Args:
          bounding_box (open3d.geometry.AxisAlignedBoundingBox) ): AxisAlignedBoundingBox to crop points
          bounding_box (open3d.geometry.OrientedBoundingBox): AxisAlignedBoundingBox to crop points

      Returns:
          open3d.geometry.PointCloud

  estimate_normals(...)
      estimate_normals(self, search_param=geometry::KDTreeSearchParamKNN with knn = 30, fast_normal_computation=True)

      Function to compute the normals of a point cloud. Normals are oriented with respect to the input point cloud if normals exist

      Args:
          search_param (open3d.geometry.KDTreeSearchParam, optional, default=geometry::KDTreeSearchParamKNN with knn = 30): The KDTree search parameters for neighborhood search.
          fast_normal_computation (bool, optional, default=True): If true, the normal estiamtion uses a non-iterative method to extract the eigenvector from the covariance matrix. This is faster, but is not as numerical stable.

      Returns:
          bool

  has_colors(...)
      has_colors(self)

      Returns ``True`` if the point cloud contains point colors.

      Returns:
          bool

  has_normals(...)
      has_normals(self)

      Returns ``True`` if the point cloud contains point normals.

      Returns:
          bool

  has_points(...)
      has_points(self)

      Returns ``True`` if the point cloud contains points.

      Returns:
          bool

  hidden_point_removal(...)
      hidden_point_removal(self, camera_location, radius)

      Removes hidden points from a point cloud and returns a mesh of the remaining points. Based on Katz et al. 'Direct Visibility of Point Sets', 2007. Additional information about the choice of radius for noisy point clouds can be found in Mehra et. al. 'Visibility of Noisy Point Cloud Data', 2010.

      Args:
          camera_location (numpy.ndarray[float64[3, 1]]): All points not visible from that location will be reomved
          radius (float): The radius of the sperical projection

      Returns:
          Tuple[open3d.geometry.TriangleMesh, List[int]]

  normalize_normals(...)
      normalize_normals(self)

      Normalize point normals to length 1.

      Returns:
          open3d.geometry.PointCloud

  orient_normals_to_align_with_direction(...)
      orient_normals_to_align_with_direction(self, orientation_reference=array([0., 0., 1.]))

      Function to orient the normals of a point cloud

      Args:
          orientation_reference (numpy.ndarray[float64[3, 1]], optional, default=array([0., 0., 1.])): Normals are oriented with respect to orientation_reference.

      Returns:
          bool

  orient_normals_towards_camera_location(...)
      orient_normals_towards_camera_location(self, camera_location=array([0., 0., 0.]))

      Function to orient the normals of a point cloud

      Args:
          camera_location (numpy.ndarray[float64[3, 1]], optional, default=array([0., 0., 0.])): Normals are oriented with towards the camera_location.

      Returns:
          bool

  paint_uniform_color(...)
      paint_uniform_color(self, color)

      Assigns each point in the PointCloud the same color.

      Args:
          color (numpy.ndarray[float64[3, 1]]): RGB color for the PointCloud.

      Returns:
          open3d.geometry.PointCloud

  remove_non_finite_points(...)
      remove_non_finite_points(self, remove_nan=True, remove_infinite=True)

      Function to remove non-finite points from the PointCloud

      Args:
          remove_nan (bool, optional, default=True): Remove NaN values from the PointCloud
          remove_infinite (bool, optional, default=True): Remove infinite values from the PointCloud

      Returns:
          open3d.geometry.PointCloud

  remove_radius_outlier(...)
      remove_radius_outlier(self, nb_points, radius)

      Function to remove points that have less than nb_points in a given sphere of a given radius

      Args:
          nb_points (int): Number of points within the radius.
          radius (float): Radius of the sphere.

      Returns:
          Tuple[open3d.geometry.PointCloud, List[int]]

  remove_statistical_outlier(...)
      remove_statistical_outlier(self, nb_neighbors, std_ratio)

      Function to remove points that are further away from their neighbors in average

      Args:
          nb_neighbors (int): Number of neighbors around the target point.
          std_ratio (float): Standard deviation ratio.

      Returns:
          Tuple[open3d.geometry.PointCloud, List[int]]

  segment_plane(...)
      segment_plane(self, distance_threshold, ransac_n, num_iterations)

      Segments a plane in the point cloud using the RANSAC algorithm.

      Args:
          distance_threshold (float): Max distance a point can be from the plane model, and still be considered an inlier.
          ransac_n (int): Number of initial points to be considered inliers in each iteration.
          num_iterations (int): Number of iterations.

      Returns:
          Tuple[numpy.ndarray[float64[4, 1]], List[int]]

  select_by_index(...)
      select_by_index(self, indices, invert=False)

      Function to select points from input pointcloud into output pointcloud.

      Args:
          indices (List[int]): Indices of points to be selected.
          invert (bool, optional, default=False): Set to ``True`` to invert the selection of indices.

      Returns:
          open3d.geometry.PointCloud

  uniform_down_sample(...)
      uniform_down_sample(self, every_k_points)

      Function to downsample input pointcloud into output pointcloud uniformly. The sample is performed in the order of the points with the 0-th point always chosen, not at random.

      Args:
          every_k_points (int): Sample rate, the selected point indices are [0, k, 2k, ...]

      Returns:
          open3d.geometry.PointCloud

  voxel_down_sample(...)
      voxel_down_sample(self, voxel_size)

      Function to downsample input pointcloud into output pointcloud with a voxel. Normals and colors are averaged if they exist.

      Args:
          voxel_size (float): Voxel size to downsample into.

      Returns:
          open3d.geometry.PointCloud

  voxel_down_sample_and_trace(...)
      voxel_down_sample_and_trace(self, voxel_size, min_bound, max_bound, approximate_class=False)

      Function to downsample using geometry.PointCloud.VoxelDownSample. Also records point cloud index before downsampling

      Args:
          voxel_size (float): Voxel size to downsample into.
          min_bound (numpy.ndarray[float64[3, 1]]): Minimum coordinate of voxel boundaries
          max_bound (numpy.ndarray[float64[3, 1]]): Maximum coordinate of voxel boundaries
          approximate_class (bool, optional, default=False)

      Returns:
          Tuple[open3d.geometry.PointCloud, numpy.ndarray[int32[m, n]], List[open3d.utility.IntVector]]

  
  Static methods defined here:

  create_from_depth_image(...) from builtins.PyCapsule
      create_from_depth_image(depth, intrinsic, extrinsic=(with default value), depth_scale=1000.0, depth_trunc=1000.0, stride=1, project_valid_depth_only=True)

      Factory function to create a pointcloud from a depth image and a
              camera. Given depth value d at (u, v) image coordinate, the corresponding 3d
              point is:

                    - z = d / depth_scale
                    - x = (u - cx) * z / fx
                    - y = (v - cy) * z / fy

      Args:
          depth (open3d.geometry.Image): The input depth image can be either a float image, or a uint16_t image.
          intrinsic (open3d.camera.PinholeCameraIntrinsic): Intrinsic parameters of the camera.
          extrinsic (numpy.ndarray[float64[4, 4]], optional) Default value:

              array([[1., 0., 0., 0.],
              [0., 1., 0., 0.],
              [0., 0., 1., 0.],
              [0., 0., 0., 1.]])
          depth_scale (float, optional, default=1000.0): The depth is scaled by 1 / depth_scale.
          depth_trunc (float, optional, default=1000.0): Truncated at depth_trunc distance.
          stride (int, optional, default=1): Sampling factor to support coarse point cloud extraction.
          project_valid_depth_only (bool, optional, default=True)

      Returns:
          open3d.geometry.PointCloud

  create_from_rgbd_image(...) from builtins.PyCapsule
      create_from_rgbd_image(image, intrinsic, extrinsic=(with default value), project_valid_depth_only=True)

      Factory function to create a pointcloud from an RGB-D image and a        camera. Given depth value d at (u, v) image coordinate, the corresponding 3d point is: - z = d / depth_scale
                    - x = (u - cx) * z / fx
                    - y = (v - cy) * z / fy

      Args:
          image (open3d.geometry.RGBDImage): The input image.
          intrinsic (open3d.camera.PinholeCameraIntrinsic): Intrinsic parameters of the camera.
          extrinsic (numpy.ndarray[float64[4, 4]], optional) Default value:

              array([[1., 0., 0., 0.],
              [0., 1., 0., 0.],
              [0., 0., 1., 0.],
              [0., 0., 0., 1.]])
          project_valid_depth_only (bool, optional, default=True)

      Returns:
          open3d.geometry.PointCloud

  
  Data descriptors defined here:

  colors
      ``float64`` array of shape ``(num_points, 3)``, range ``[0, 1]`` , use ``numpy.asarray()`` to access data: RGB colors of points.

  normals
      ``float64`` array of shape ``(num_points, 3)``, use ``numpy.asarray()`` to access data: Points normals.

  points
      ``float64`` array of shape ``(num_points, 3)``, use ``numpy.asarray()`` to access data: Points coordinates.

  
  Methods inherited from Geometry3D:

  get_axis_aligned_bounding_box(...)
      get_axis_aligned_bounding_box(self)

      Returns an axis-aligned bounding box of the geometry.

      Returns:
          open3d.geometry.AxisAlignedBoundingBox

  get_center(...)
      get_center(self)

      Returns the center of the geometry coordinates.

      Returns:
          numpy.ndarray[float64[3, 1]]

  get_max_bound(...)
      get_max_bound(self)

      Returns max bounds for geometry coordinates.

      Returns:
          numpy.ndarray[float64[3, 1]]

  get_min_bound(...)
      get_min_bound(self)

      Returns min bounds for geometry coordinates.

      Returns:
          numpy.ndarray[float64[3, 1]]

  get_oriented_bounding_box(...)
      get_oriented_bounding_box(self)

      Returns an oriented bounding box of the geometry.

      Returns:
          open3d.geometry.OrientedBoundingBox

  rotate(...)
      rotate(self, R, center=True)

      Apply rotation to the geometry coordinates and normals.

      Args:
          R (numpy.ndarray[float64[3, 3]]): The rotation matrix
          center (bool, optional, default=True): If true, then the rotation is applied to the centered geometry

      Returns:
          open3d.geometry.Geometry3D

  scale(...)
      scale(self, scale, center=True)

      Apply scaling to the geometry coordinates.

      Args:
          scale (float): The scale parameter that is multiplied to the points/vertices of the geometry
          center (bool, optional, default=True): If true, then the scale is applied to the centered geometry

      Returns:
          open3d.geometry.Geometry3D

  transform(...)
      transform(self, arg0)

      Apply transformation (4x4 matrix) to the geometry coordinates.

      Args:
          arg0 (numpy.ndarray[float64[4, 4]])

      Returns:
          open3d.geometry.Geometry3D

  translate(...)
      translate(self, translation, relative=True)

      Apply translation to the geometry coordinates.

      Args:
          translation (numpy.ndarray[float64[3, 1]]): A 3D vector to transform the geometry
          relative (bool, optional, default=True): If true, the translation vector is directly added to the geometry coordinates. Otherwise, the center is moved to the translation vector.

      Returns:
          open3d.geometry.Geometry3D

  
  Static methods inherited from Geometry3D:

  get_rotation_matrix_from_axis_angle(...) from builtins.PyCapsule
      get_rotation_matrix_from_axis_angle(rotation: numpy.ndarray[float64[3, 1]]) -> numpy.ndarray[float64[3, 3]]

  get_rotation_matrix_from_quaternion(...) from builtins.PyCapsule
      get_rotation_matrix_from_quaternion(rotation: numpy.ndarray[float64[4, 1]]) -> numpy.ndarray[float64[3, 3]]

  get_rotation_matrix_from_xyz(...) from builtins.PyCapsule
      get_rotation_matrix_from_xyz(rotation: numpy.ndarray[float64[3, 1]]) -> numpy.ndarray[float64[3, 3]]

  get_rotation_matrix_from_xzy(...) from builtins.PyCapsule
      get_rotation_matrix_from_xzy(rotation: numpy.ndarray[float64[3, 1]]) -> numpy.ndarray[float64[3, 3]]

  get_rotation_matrix_from_yxz(...) from builtins.PyCapsule
      get_rotation_matrix_from_yxz(rotation: numpy.ndarray[float64[3, 1]]) -> numpy.ndarray[float64[3, 3]]

  get_rotation_matrix_from_yzx(...) from builtins.PyCapsule
      get_rotation_matrix_from_yzx(rotation: numpy.ndarray[float64[3, 1]]) -> numpy.ndarray[float64[3, 3]]

  get_rotation_matrix_from_zxy(...) from builtins.PyCapsule
      get_rotation_matrix_from_zxy(rotation: numpy.ndarray[float64[3, 1]]) -> numpy.ndarray[float64[3, 3]]

  get_rotation_matrix_from_zyx(...) from builtins.PyCapsule
      get_rotation_matrix_from_zyx(rotation: numpy.ndarray[float64[3, 1]]) -> numpy.ndarray[float64[3, 3]]

  
  Methods inherited from Geometry:

  clear(...)
      clear(self)

      Clear all elements in the geometry.

      Returns:
          open3d.geometry.Geometry

  dimension(...)
      dimension(self)

      Returns whether the geometry is 2D or 3D.

      Returns:
          int

  get_geometry_type(...)
      get_geometry_type(self)

      Returns one of registered geometry types.

      Returns:
          open3d.geometry.Geometry.GeometryType

  is_empty(...)
      is_empty(self)

      Returns ``True`` iff the geometry is empty.

      Returns:
          bool

  
  Data and other attributes inherited from Geometry:

  HalfEdgeTriangleMesh = Type.HalfEdgeTriangleMesh

  Image = Type.Image

  LineSet = Type.LineSet

  PointCloud = Type.PointCloud

  RGBDImage = Type.RGBDImage

  TetraMesh = Type.TetraMesh

  TriangleMesh = Type.TriangleMesh

  Type = <class 'open3d.open3d_pybind.geometry.Geometry.Type'>
      Enum class for Geometry types.

  Unspecified = Type.Unspecified

  VoxelGrid = Type.VoxelGrid

  
  Methods inherited from pybind11_builtins.pybind11_object:

  _new_(*args, **kwargs) from pybind11_builtins.pybind11_type
      Create and return a new object.  See help(type) for accurate signatureo3d.visualization.draw,
                          geometry=geometry,
                          title=title,
                          width=width,
                          height=height,
                          actions=actions,
                          lookat=lookat,
                          eye=eye,
                          up=up,
                          field_of_view=field_of_view,
                          bg_color=bg_color,
                          bg_image=bg_image,
                          show_ui=show_ui,
                          point_size=point_size,
                          animation_time_step=animation_time_step,
                          animation_duration=animation_duration,
                          rpc_interface=rpc_interface,
                          on_init=on_init,
                          on_animation_frame=on_animation_frame,
                          on_animation_tick=on_animation_tick,
                          non_blocking_and_return_uid=True))
    visualizer = WebVisualizer(window_uid=window_uid)
    visualizer.show()
