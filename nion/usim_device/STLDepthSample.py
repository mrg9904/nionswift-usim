from __future__ import annotations

import math
import pathlib
import typing

import numpy
import numpy.typing
import trimesh

from nion.usim_device import SampleSimulator
from nion.usim_device import SimulationSettings
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

        self.__surface_cache: typing.Optional[
            typing.Tuple[
                _NDArray,
                _NDArray,
            ]
        ] = None

        # Cache the generated depth planes separately because their spacing
        # also depends on the current FOV-derived slice thickness.
        self.__depth_cache_key: typing.Any = None

        self.__depth_cache: typing.Optional[
            typing.List[
                typing.Tuple[
                    float,
                    _NDArray,
                ]
            ]
        ] = None

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

    def __first_hit_z(
        self,
        ray_origins: numpy.typing.NDArray[numpy.float64],
        direction_z: float,
    ) -> numpy.typing.NDArray[numpy.float64]:
        """Return the first intersection Z coordinate for every ray.

        Rays that do not intersect the mesh retain a NaN value.
        """

        hit_z_nm = numpy.full(
            ray_origins.shape[0],
            numpy.nan,
            dtype=numpy.float64,
        )

        chunk_size = max(
            1,
            int(
                SimulationSettings.STL_RAY_CHUNK_SIZE
            ),
        )

        for start in range(
            0,
            ray_origins.shape[0],
            chunk_size,
        ):
            stop = min(
                start + chunk_size,
                ray_origins.shape[0],
            )

            origins = ray_origins[start:stop]

            directions = numpy.zeros_like(origins)
            directions[:, 2] = direction_z

            locations, ray_indices, _ = (
                self.__intersector.intersects_location(
                    ray_origins=origins,
                    ray_directions=directions,
                    multiple_hits=False,
                )
            )

            if ray_indices.size:
                hit_z_nm[
                    start + ray_indices
                ] = locations[:, 2]

        return hit_z_nm

    def __calculate_surface_maps(
        self,
        offset_m: Geometry.FloatPoint,
        fov_size_nm: Geometry.FloatSize,
        extra_nm: Geometry.FloatPoint,
        center_nm: Geometry.FloatPoint,
        used_size: Geometry.IntSize,
    ) -> typing.Tuple[_NDArray, _NDArray]:
        """Calculate lower and upper sample surfaces at every XY pixel."""

        geometry_key = self.__geometry_key(
            offset_m,
            fov_size_nm,
            extra_nm,
            center_nm,
            used_size,
        )

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

        xx_nm, yy_nm = numpy.meshgrid(
            x_nm,
            y_nm,
        )

        ray_count = (
            used_size.height
            * used_size.width
        )

        # Generate one downward ray from above the entire STL model.
        upper_origins = numpy.empty(
            (ray_count, 3),
            dtype=numpy.float64,
        )

        upper_origins[:, 0] = xx_nm.ravel()
        upper_origins[:, 1] = yy_nm.ravel()

        upper_origins[:, 2] = (
            self.__mesh.bounds[1, 2]
            + 1.0
        )

        # Generate one upward ray from below the entire STL model.
        lower_origins = upper_origins.copy()

        lower_origins[:, 2] = (
            self.__mesh.bounds[0, 2]
            - 1.0
        )

        upper_surface_nm = self.__first_hit_z(
            upper_origins,
            direction_z=-1.0,
        ).reshape(
            used_size.height,
            used_size.width,
        )

        lower_surface_nm = self.__first_hit_z(
            lower_origins,
            direction_z=1.0,
        ).reshape(
            used_size.height,
            used_size.width,
        )

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
    ) -> typing.List[
        typing.Tuple[
            float,
            _NDArray,
        ]
    ]:
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
            self.__depth_cache = depth_planes

            return depth_planes

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
        self.__depth_cache = depth_planes

        return depth_planes