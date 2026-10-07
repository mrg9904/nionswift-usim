from __future__ import annotations

import math
import pathlib
import typing
import logging

import numpy
import numpy.typing
import trimesh

from nion.usim_device import SampleSimulator
from nion.usim_device import SimulationSettings
from nion.usim_device import SurfaceRasterizer
from nion.usim_device import HAADFFocusModel
from nion.utils import Geometry


_NDArray = numpy.typing.NDArray[numpy.float32]


class STLDepthSample(SampleSimulator.Sample):
    """A depth-resolved HAADF sample loaded from a watertight STL mesh.

    The STL must be column-solid along Z: for each XY point, material must
    occupy either zero or one continuous Z interval. The generated cuboid and
    side-sphere model satisfies this requirement.
    """

    def __init__(self, stage_size_nm: float) -> None:
        # Keep the standard uSim sample constructor interface.
        _ = stage_size_nm

        stl_path = (
            pathlib.Path(__file__).resolve().parent
            / "samples"
            / SimulationSettings.STL_SAMPLE_FILE_NAME
        )

        if not stl_path.is_file():
            raise FileNotFoundError(
                f"STL sample file was not found: {stl_path}"
            )

        # The numerical coordinates stored in the STL are interpreted
        # directly as nanometers.
        mesh = trimesh.load_mesh(
            stl_path,
            file_type="stl",
            force="mesh",
            process=True,
        )

        if not isinstance(mesh, trimesh.Trimesh):
            raise RuntimeError(
                f"The STL did not load as a triangle mesh: {stl_path}"
            )

        if not mesh.is_watertight:
            raise RuntimeError(
                "The STL sample must be watertight for depth slicing"
            )

        # Move the selected cuboid to the default uSim scan center.
        mesh.apply_translation(
            (
                SimulationSettings.STL_SAMPLE_SHIFT_X_NM,
                SimulationSettings.STL_SAMPLE_SHIFT_Y_NM,
                0.0,
            )
        )

        self.__mesh = mesh
        self.__rasterizer = SurfaceRasterizer.SurfaceRasterizer(mesh.triangles)

        # RayMeshIntersector uses an R-tree spatial index to avoid checking
        # every ray against every triangle in the mesh.
        self.__intersector = (
            trimesh.ray.ray_triangle.RayMeshIntersector(
                self.__mesh
            )
        )

        # This initial implementation provides HAADF geometry only.
        # No EELS features are assigned yet.
        self.__features: typing.List[
            SampleSimulator.Feature
        ] = list()

        # Cache the expensive ray intersections. Changing defocus does not
        # invalidate this cache because defocus does not change the geometry.
        self.__surface_cache_key: typing.Any = None

        self.__surface_cache: typing.Optional[typing.Tuple[typing.Any, typing.Any]] = None

        # Cache the generated depth planes separately because their spacing
        # also depends on the current FOV-derived slice thickness.
        self.__depth_cache_key: typing.Any = None

        self.__depth_cache: typing.Optional[typing.Sequence[typing.Tuple[float, _NDArray]]] = None
        self.__gpu_depth_enabled = True

    @property
    def title(self) -> str:
        """Name shown in the uSim sample selector."""

        return "STL Depth Sample"

    @property
    def features(
        self,
    ) -> typing.List[SampleSimulator.Feature]:
        """Return EELS-compatible features.

        The initial STL implementation only provides HAADF geometry, so this
        list is empty.
        """

        return self.__features

    @staticmethod
    def __geometry_key(
        offset_m: Geometry.FloatPoint,
        fov_size_nm: Geometry.FloatSize,
        extra_nm: Geometry.FloatPoint,
        center_nm: Geometry.FloatPoint,
        used_size: Geometry.IntSize,
    ) -> typing.Tuple[typing.Any, ...]:
        """Create a cache key describing the current scan geometry."""

        return (
            offset_m.as_tuple(),
            fov_size_nm.as_tuple(),
            extra_nm.as_tuple(),
            center_nm.as_tuple(),
            used_size.as_tuple(),
        )

    def __surface_hits_from_above(
        self,
        ray_origins: numpy.typing.NDArray[numpy.float64],
    ) -> typing.Tuple[
        numpy.typing.NDArray[numpy.float64],
        numpy.typing.NDArray[numpy.float64],
    ]:
        """Return lower and upper surface Z values using one ray pass."""

        ray_count = ray_origins.shape[0]

        lower_surface_nm = numpy.full(
            ray_count,
            numpy.inf,
            dtype=numpy.float64,
        )

        upper_surface_nm = numpy.full(
            ray_count,
            -numpy.inf,
            dtype=numpy.float64,
        )

        chunk_size = max(
            1,
            int(SimulationSettings.STL_RAY_CHUNK_SIZE),
        )

        for start in range(
            0,
            ray_count,
            chunk_size,
        ):
            stop = min(
                start + chunk_size,
                ray_count,
            )

            origins = ray_origins[start:stop]

            directions = numpy.zeros_like(origins)
            directions[:, 2] = -1.0

            locations, ray_indices, _ = (
                self.__intersector.intersects_location(
                    ray_origins=origins,
                    ray_directions=directions,
                    multiple_hits=True,
                )
            )

            if not ray_indices.size:
                continue

            global_ray_indices = (
                start + ray_indices
            )

            hit_z_nm = locations[:, 2]

            numpy.minimum.at(
                lower_surface_nm,
                global_ray_indices,
                hit_z_nm,
            )

            numpy.maximum.at(
                upper_surface_nm,
                global_ray_indices,
                hit_z_nm,
            )

        no_intersection = (
            ~numpy.isfinite(lower_surface_nm)
            | ~numpy.isfinite(upper_surface_nm)
        )

        lower_surface_nm[no_intersection] = numpy.nan
        upper_surface_nm[no_intersection] = numpy.nan

        return (
            lower_surface_nm,
            upper_surface_nm,
        )

    def __calculate_surface_maps(
        self,
        offset_m: Geometry.FloatPoint,
        fov_size_nm: Geometry.FloatSize,
        extra_nm: Geometry.FloatPoint,
        center_nm: Geometry.FloatPoint,
        used_size: Geometry.IntSize,
        *, on_gpu: bool = False,
    ) -> typing.Tuple[typing.Any, typing.Any]:
        """Calculate lower and upper sample surfaces at every XY pixel."""

        geometry_key = (on_gpu, self.__geometry_key(
            offset_m,
            fov_size_nm,
            extra_nm,
            center_nm,
            used_size,
        ))

        if (
            geometry_key == self.__surface_cache_key
            and self.__surface_cache is not None
        ):
            return self.__surface_cache

        total_fov_nm = fov_size_nm + extra_nm

        # Follow the coordinate convention already used by the original
        # uSim sample classes.
        scan_center_x_nm = (
            center_nm.x
            - offset_m.x * 1e9
        )

        scan_center_y_nm = (
            center_nm.y
            - offset_m.y * 1e9
        )

        left_nm = (
            scan_center_x_nm
            - total_fov_nm.width * 0.5
        )

        top_nm = (
            scan_center_y_nm
            - total_fov_nm.height * 0.5
        )

        pixel_width_nm = (
            total_fov_nm.width
            / used_size.width
        )

        pixel_height_nm = (
            total_fov_nm.height
            / used_size.height
        )

        # Use pixel-center coordinates rather than pixel-edge coordinates.
        x_nm = (
            left_nm
            + (
                numpy.arange(
                    used_size.width,
                    dtype=numpy.float64,
                )
                + 0.5
            )
            * pixel_width_nm
        )

        y_nm = (
            top_nm
            + (
                numpy.arange(
                    used_size.height,
                    dtype=numpy.float64,
                )
                + 0.5
            )
            * pixel_height_nm
        )

        if SimulationSettings.STL_USE_SURFACE_RASTERIZER:
            lower_surface_nm, upper_surface_nm = self.__rasterizer.surface_maps(x_nm, y_nm, on_gpu=on_gpu)
        else:
            # Retain the original ray path for numerical comparisons.
            xx_nm, yy_nm = numpy.meshgrid(x_nm, y_nm)
            ray_origins = numpy.empty((xx_nm.size, 3), dtype=numpy.float64)
            ray_origins[:, 0] = xx_nm.ravel()
            ray_origins[:, 1] = yy_nm.ravel()
            ray_origins[:, 2] = self.__mesh.bounds[1, 2] + 1.0
            lower_flat, upper_flat = self.__surface_hits_from_above(ray_origins)
            lower_surface_nm = lower_flat.reshape(used_size.height, used_size.width)
            upper_surface_nm = upper_flat.reshape(used_size.height, used_size.width)

        upper_surface_nm = (
            upper_surface_nm.astype(
                numpy.float32,
                copy=False,
            )
        )

        lower_surface_nm = (
            lower_surface_nm.astype(
                numpy.float32,
                copy=False,
            )
        )

        self.__surface_cache_key = geometry_key

        self.__surface_cache = (
            lower_surface_nm,
            upper_surface_nm,
        )

        # A changed scan geometry always invalidates the depth-plane cache.
        self.__depth_cache_key = None
        self.__depth_cache = None

        return self.__surface_cache

    def plot_features(
        self,
        data: _NDArray,
        offset_m: Geometry.FloatPoint,
        fov_size_nm: Geometry.FloatSize,
        extra_nm: Geometry.FloatPoint,
        center_nm: Geometry.FloatPoint,
        used_size: Geometry.IntSize,
    ) -> None:
        """Generate an unfocused two-dimensional thickness projection.

        The HAADF scan path does not normally call this method for the STL
        sample because it uses generate_depth_planes directly. This method is
        retained to satisfy the Sample interface and support other callers.
        """

        lower_surface_nm, upper_surface_nm = (
            self.__calculate_surface_maps(
                offset_m,
                fov_size_nm,
                extra_nm,
                center_nm,
                used_size,
            )
        )

        valid = (
            numpy.isfinite(lower_surface_nm)
            & numpy.isfinite(upper_surface_nm)
        )

        thickness_nm = numpy.zeros_like(
            data,
            dtype=numpy.float32,
        )

        thickness_nm[valid] = numpy.maximum(
            upper_surface_nm[valid]
            - lower_surface_nm[valid],
            0.0,
        )

        data += (
            thickness_nm
            / SimulationSettings.STL_REFERENCE_THICKNESS_NM
        )

    def generate_depth_planes(
        self,
        offset_m: Geometry.FloatPoint,
        fov_size_nm: Geometry.FloatSize,
        extra_nm: Geometry.FloatPoint,
        center_nm: Geometry.FloatPoint,
        used_size: Geometry.IntSize,
        slice_thickness_nm: float,
    ) -> typing.Sequence[typing.Tuple[float, _NDArray]]:
        """Convert the STL volume into uSim depth planes.

        Each returned item contains:

            (depth_plane_center_nm, HAADF_contribution)

        The intensity contribution is proportional to the amount of material
        contained in that depth slice.
        """

        if slice_thickness_nm <= 0.0:
            raise ValueError(
                "slice_thickness_nm must be greater than zero"
            )

        geometry_key = self.__geometry_key(
            offset_m,
            fov_size_nm,
            extra_nm,
            center_nm,
            used_size,
        )

        depth_key = (
            geometry_key,
            SimulationSettings.STL_NORMALIZE_COLUMN_INTENSITY,
            SimulationSettings.STL_REFERENCE_THICKNESS_NM,
            SimulationSettings.STL_NORMALIZED_COLUMN_INTENSITY,
            round(
                slice_thickness_nm,
                12,
            ),
        )

        if (
            depth_key == self.__depth_cache_key
            and self.__depth_cache is not None
        ):
            return self.__depth_cache

        if (
            self.__gpu_depth_enabled
            and SimulationSettings.STL_USE_SURFACE_RASTERIZER
            and self.__rasterizer.uses_gpu((used_size.height, used_size.width))
        ):
            try:
                from nion.usim_device.GPUHAADF import GPUPreparedDepthPlanes
                lower, upper = self.__calculate_surface_maps(
                    offset_m, fov_size_nm, extra_nm, center_nm, used_size, on_gpu=True,
                )
                self.__depth_cache = GPUPreparedDepthPlanes(lower, upper, slice_thickness_nm)
                self.__depth_cache_key = depth_key
                return self.__depth_cache
            except Exception:
                self.__gpu_depth_enabled = False
                logging.warning("uSim GPU depth preparation failed; using CPU", exc_info=True)

        lower_surface_nm, upper_surface_nm = (
            self.__calculate_surface_maps(
                offset_m,
                fov_size_nm,
                extra_nm,
                center_nm,
                used_size,
            )
        )

        valid = (
            numpy.isfinite(lower_surface_nm)
            & numpy.isfinite(upper_surface_nm)
            & (
                upper_surface_nm
                > lower_surface_nm
            )
        )

        # If the current FOV contains no part of the STL, return one empty
        # plane. apply_depth_planes_defocus requires at least one plane.
        if not numpy.any(valid):
            empty_plane = numpy.zeros(
                (
                    used_size.height,
                    used_size.width,
                ),
                dtype=numpy.float32,
            )

            depth_planes = [
                (
                    0.0,
                    empty_plane,
                )
            ]

            self.__depth_cache_key = depth_key
            self.__depth_cache = HAADFFocusModel.PreparedDepthPlanes(depth_planes)

            return self.__depth_cache

        minimum_depth_nm = float(
            numpy.min(
                lower_surface_nm[valid]
            )
        )

        maximum_depth_nm = float(
            numpy.max(
                upper_surface_nm[valid]
            )
        )

        # Anchor the depth-slice grid at z = 0 nm. This makes depth-plane
        # positions reproducible when the FoV moves laterally.
        first_slice_index = math.floor(
            minimum_depth_nm
            / slice_thickness_nm
        )

        final_slice_index = math.ceil(
            maximum_depth_nm
            / slice_thickness_nm
        )

        depth_planes: typing.List[
            typing.Tuple[
                float,
                _NDArray,
            ]
        ] = list()

        for slice_index in range(
            first_slice_index,
            final_slice_index,
        ):
            lower_slice_nm = (
                slice_index
                * slice_thickness_nm
            )

            upper_slice_nm = (
                lower_slice_nm
                + slice_thickness_nm
            )

            # Calculate the material thickness contained in this slice at
            # every XY pixel:
            #
            # overlap = min(sample_top, slice_top)
            #           - max(sample_bottom, slice_bottom)
            overlap_nm = numpy.minimum(
                upper_surface_nm,
                upper_slice_nm,
            ) - numpy.maximum(
                lower_surface_nm,
                lower_slice_nm,
            )

            overlap_nm = numpy.where(
                valid,
                numpy.clip(
                    overlap_nm,
                    0.0,
                    slice_thickness_nm,
                ),
                0.0,
            ).astype(
                numpy.float32,
                copy=False,
            )

            # Do not retain completely empty depth planes.
            if not numpy.any(
                overlap_nm > 0.0
            ):
                continue

            plane_data = (
                overlap_nm
                / SimulationSettings.STL_REFERENCE_THICKNESS_NM
            ).astype(
                numpy.float32,
                copy=False,
            )

            center_depth_nm = (
                lower_slice_nm
                + upper_slice_nm
            ) * 0.5

            depth_planes.append(
                (
                    center_depth_nm,
                    plane_data,
                )
            )

        # Defensive fallback for numerical edge cases.
        if not depth_planes:
            depth_planes.append(
                (
                    0.0,
                    numpy.zeros(
                        (
                            used_size.height,
                            used_size.width,
                        ),
                        dtype=numpy.float32,
                    ),
                )
            )
        # Optional diagnostic normalization.
        #
        # Before normalization:
        #     summed intensity is proportional to projected thickness.
        #
        # After normalization:
        #     every occupied XY pixel has the same total ideal intensity,
        #     while its relative intensity distribution along Z is retained.
        if SimulationSettings.STL_NORMALIZE_COLUMN_INTENSITY:
            total_projection = numpy.zeros(
                (
                    used_size.height,
                    used_size.width,
                ),
                dtype=numpy.float32,
            )

            for _, plane_data in depth_planes:
                total_projection += plane_data

            occupied = total_projection > 0.0

            column_scale = numpy.zeros_like(
                total_projection,
                dtype=numpy.float32,
            )

            numpy.divide(
                SimulationSettings.STL_NORMALIZED_COLUMN_INTENSITY,
                total_projection,
                out=column_scale,
                where=occupied,
            )

            for _, plane_data in depth_planes:
                plane_data *= column_scale
        
        self.__depth_cache_key = depth_key
        self.__depth_cache = HAADFFocusModel.PreparedDepthPlanes(depth_planes)

        return self.__depth_cache
