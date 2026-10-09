Nion Swift STEM Microscope Simulator
====================================

The Nion Swift STEM Microscope Simulator Library (used in Nion Swift)
---------------------------------------------------------------------
A STEM microscope simulator for use with Nion Swift. Used for debugging Nion Swift acquisition and developing acquisition tools, techniques, and apps.

.. start-badges

.. list-table::
    :stub-columns: 1

    * - tests
      - | |linux|
    * - package
      - |version|


.. |linux| image:: https://img.shields.io/travis/nion-software/nionswift-usim/master.svg?label=Linux%20build
   :target: https://travis-ci.org/nion-software/nionswift-usim
   :alt: Travis CI build status (Linux)

.. |version| image:: https://img.shields.io/pypi/v/nionswift-usim.svg
   :target: https://pypi.org/project/nionswift-usim/
   :alt: Latest PyPI version

.. end-badges

Introduction
------------
Microscopes come in many varieties such as light/optical microscopes, electron microscopes, scanning probe microscopes. Within each of those broad categories are many sub-categories such as fluorescence and super resolution optical microscopes, and transmission or scanning electron microscopes.

This simulator is intended to provide a platform and specific simulations for a variety of microscopes.

Currently, however, it is primarily focused on scanning transmission electron microscopy (STEM). A STEM instrument produces electrons which are transmitted through a sample, scattering, deflecting, or changing energy in the process, and then digitized in a detector.

A STEM instrument will include an electron source, electromagnetic lenses to form and control the electron beam, a scanning system to control the lenses, and multiple cameras to detect the resulting electrons. A typical configuration of detectors would include multiple annular dark field detectors, high-angle HAADF, medium-angle MAADF; a Ronchigram detector (an image of the convergent beam diffraction pattern, useful and important in aberration corrected instruments); often an EELS detector (electron energy loss spectrum) to measure energy loss; and video cameras for alignment and adjustment. Other detectors such as optical or x-ray detectors are also sometimes present.

In transmission electron microscopy, as an electron travels through a sample, it may interact with electrons or nuclei, scattering in the process. An ideal sample for transmission electron microscopy will be thin, to limit the number of scattering events. As it scatters, it may transfer some of its energy to the sample (inelastic) or simply be deflected (elastic). When it transfers energy to the sample (inelastic), the amount of energy that the electron loses will be characteristic (on average) of the atom or compound with which it interacts. This may show up in EELS data as phonons (interaction with nuclei), plasmons (interaction with surfaces), or edges (interactions with electrons of specific energies).

.. describe applications

Features
--------
The STEM microscope simulator can simulate a HAADF detector within a beam scan, a Ronchigram with lens aberrations, and a 2D EELS detector that can be summed in the zero dimension to produce a 1D EELS spectrum.

The EELS detector produces EELS spectra unique to each beam position within the simulated sample. The Ronchigram is not dependent on beam position.

The simulated sample can be basic elements/vacuum arranged in rectangular regions; or an amorphous sample.

The scan and Ronchigram or EELS detector can be combined to do spectrum imaging.

The simulator has been used to develop many acquisition algorithms and even some tuning and alignment algorithms for use on the Nion STEM instrument.

Acquisition Modes
-----------------
There are several regularly used modes of acquisition on a STEM instrument.

To begin, an imaging mode formed by continuously scanning the beam in a rectangular pattern with a HAADF detector is often the mode used to find the beam and make rough adjustments. At each point of the scan, a single value is read from the HAADF detector and those values are assembled into an image. The scan size can be adjusted by changing the field of view (FOV), expressed in nm; and also rotated to align with a sample. The pixel time can also be adjusted to account for a weaker or stronger signal, with the tradeoff being the speed of imaging.

Once the beam is centered, the Ronchigram imaging mode is used to focus and correct aberrations in the electromagnetic lenses. The magnification on the Ronchigram can be adjusted using the defocus value, expressed in nm. When the beam has no defocus (0nm), it would appear as an image with little or no contrast, depending on the sample and other alignments. When the beam has a slight defocus (500nm), it forms an image of the sample and can be used for imaging. However, lens aberrations may contribute distortions to the resulting image. This simulator provides simulator aberrations up to 5th order. The default values produce a perfectly aligned instrument; but explicitly introducing aberrations can be useful for learning about aberrations or testing alignment/tuning software.

In a STEM instrument with a spectrometer, an electron energy loss spectrum (EELS) can also be observed. The spectrometer is a special set of electromagnetic lenses that bend the electrons in such a way that the energy of the electron is dispersed in a spatial axis, forming an image where the vertical axis is a non-dimensioned spread of the beam and the horizontal axis is an energy loss for the electron. The energy loss will be characteristic (statistically) of the atom or compound through which the electron travelled.

The simulator provides a 2D EELS spectrum that is calculated based on the beam position (also known as the probe position) on the sample. The 2D EELS spectrum can be summed along the vertical axis to produce a 1D EELS spectrum that can be displayed as a line plot. The probe position can be adjusted by starting a scan and then stopping it, enabling the beam position graphic, and adjusting it.

The scan and EELS spectrum can be combined into a spectrum image. The beam is scanned and at each beam position, an EELS spectrum is acquired. The results are assembled in a data item 2D x 1D where the first two dimensions represent the scan position and the last dimension represents the electron energy loss at that position. Since each EELS spectrum is dependent on the sample at the corresponding beam position, the spectrum image can be used to do things like elemental mapping, where the concentration of one or more elements are formed into images where the intensity represents their relative concentration within the sample.

- 4D STEM: Scanning with Ronchigram detector, combined into 2D x 2D data item
- 4D STEM virtual detectors: Scanning with Ronchigram detector, summed within areas to form one or more 2D images
- Spectrum imaging, short exposure, aligned and summed

.. describe where each of these modes would be used

Samples
-------
The simulator provides two types of samples: a crystal and an amorphous sample.

.. describe each sample and why it is used

Tutorial
--------
Starting with a new or existing project, create a new workspace (**Workspace > New Workspace...**) or clear the existing one. Then split the workspace into a 2x2 layout (**Workspace > Display Panel Split > Split 2x2**).

In the top left panel, right click and choose **uSim Scan (HAADF)**. Press **Scan** at the lower left of that panel. Then press **Stop**.

Open the scan control panel if it is not already open (**Window > uSim Scan Scan Control**). In the scan control panel, click **Positioned**. This will put a probe marker on the scan image. This will only appear when the scan is stopped. You can drag it around.

In the top right panel, right click and choose **uSim EELS Camera**. This shows the 2D EELS detector. Make it display the 1D EELS by clicking the checkbox at the bottom of the panel. Then pres **Play** at the lower left of that panel.

Now drag the probe marker on the scan and the EELS data will change according to the probe position. Press **Pause** on the EELS data when finished.

Open the spectrum imaging panel if it is not already ope (**Window > Spectrum Imaging / 4d Scan Acquisition**). Choose **uSim EELS Camera** at the top left of that panel and choose **Spectra** as the acquisition type. Change the scan width to 24 and the camera exposure time to 50ms. Then click **Acquire**. This will acquire a spectrum image of the scan area and the EELS. You should see the scan from top to bottom and the resulting spectrum image. At the end, you will see the captured HAADF image.

Once the acquisition is finished, right click on the HAADF image which should be in the bottom right panel. Choose **Delete Data Item "Spectrum Image (HAADF)"**. We will not be using that data in this tutorial.

Click on the panel with the spectrum image named "Spectrum Image (uSim EELS Camera)". Then press "p". This will attach a pick region to the spectrum image and display the resulting sum of the spectra within the pick region in a new line plot display. You can drag the display interval to change the energy range displayed on the spectrum image. You can also zoom and otherwise examine the data in the EELS spectra.

Next right click on the spectrum image and choose **Delete Data Item "Spectrum Image (uSim EELS Camera)"**. This will delete the spectrum image and the associated pick line plot.

Now right click on an empty display panel and choose **uSim Ronchigram Camera**. Press **Play**.

Next open the simulator control panel **Window > Simulator Control**. In the simulator panel, you can change various parameters that will affect the Ronchigram. Try changing **C23 X** to 500 nm.

TODO:

    - 4D STEM (Spectrum Imaging using Ronchigram and Images)
    - Change sample type (top of Simulator Control)
    - Change EELS dispersion (set eV/ch to 10 in Simulator Control)
    - Find an edge in EELS data (look in 1200 - 1300 eV range)
    - Background removal (if EELS analysis installed)
    - Scan and subscan, rotation
    - Spectrum imaging with drift correction
    - Multiple shift EELS acquire
    - Multi Acquire
    - Multi Acquire SI
    - 4D STEM with masking
    - Line scan

More Information
----------------

- `Changelog <https://github.com/nion-software/nionswift-usim/blob/master/CHANGES.rst>`_

STL HAADF performance
---------------------
The STL sample evaluates the mesh surfaces directly at scan pixel centers,
avoiding a general ray-intersection query whenever the FoV changes. Near
focus, it uses a spatial Gaussian. Larger blur uses a reflected-boundary
frequency-domain Gaussian whose width continues to grow with defocus.
STL depth spectra are cached within a 64 MiB budget and combined before a
single inverse transform per image. The depth spacing and specimen geometry
are preserved. At extreme defocus the image naturally approaches its mean.

``STL_SURFACE_BACKEND="auto"`` is the default: grids of at least 512 x 512
use CUDA when available. Surfaces, depth slices, cached spectra, Gaussian
blur and detector noise remain on the GPU; the final image is transferred
to the CPU. Rotated scans transfer the ideal image before CPU rotation and
noise. GPU spectra have a separate 512 MiB cache budget. Small grids and
machines without CuPy/CUDA use the CPU path. ``"cpu"`` disables CUDA and
``"gpu"`` also enables it for small grids. Restart uSim after changes.
``STL_USE_SURFACE_RASTERIZER=False`` selects the original ray reference.

Install the optional GPU dependency in the same environment as Nion Swift::

    python -m pip install "cupy-cuda12x[ctk]"

The package also exposes this dependency as the ``gpu`` extra, for example
``python -m pip install -e ".[gpu]"`` from this checkout.

An NVIDIA driver is required. The CUDA runtime packages are installed with
the extra above. The backend has been validated on an RTX 4070 Laptop GPU.
At 1028 x 1028, after CUDA initialization, representative FoV changes take
about 39-64 ms including geometry, imaging, noise and transfer, compared
with roughly 1.2-1.6 s of CPU geometry, imaging and noise. Initial CUDA setup and
kernel compilation take longer and are cached between runs. These are
computation times; acquisition still observes the requested pixel dwell.

Run the synchronized benchmark (or use ``--backend cpu``)::

    python tools/benchmark_haadf.py --backend gpu --size 1028

HAADF uses Poisson electron-counting noise and optional detector read noise.
Mean counts depend on intensity, beam current, dwell time and detection
efficiency. Relative shot noise decreases as the inverse square root of
dwell: at the reference 200 pA, ideal intensity 1 and efficiency 0.3, it is
about 5.2% at 1 us, 2.6% at 4 us and 1.3% at 16 us. Integrated image
brightness still increases with dwell; divide by dwell to compare normalized
intensities. ``HAADF_SHOT_NOISE_ENABLED``, ``HAADF_DETECTION_EFFICIENCY`` and
``HAADF_READ_NOISE_ELECTRONS`` in ``SimulationSettings.py`` configure this
approximate detector model. Set shot noise off and read noise to zero for
deterministic images.

Run the numerical regression checks in the Nion Swift environment::

    python -m unittest nionswift_plugin.usim.test.HAADFPerformance_test -v


Geometry-driven EELS
--------------------

EELS now queries the local specimen beam-path length, rather than counting
features as fixed 30 nm layers. ``Sample.eels_layers_at`` returns thicknesses
and synthetic material parameters in absolute sample coordinates. Thickness
blocks use 20/50/100 nm; the radius 50 nm spherical sample uses the chord
``t(r) = 2 sqrt(R^2-r^2)`` and returns vacuum outside its circular projection.
STL uses the same vertical surface rasterizer as HAADF. The column-solid STL
constraint still applies. Legacy flat features without thickness retain their
30 nm default. Stage/beam offsets, scan center and scan rotation affect the
probe geometry.

The inelastic optical depth is ``tau = sum(t_i/lambda_i)``. Independent events
follow ``P(n) = exp(-tau) tau^n/n!``; each order convolves the single-event
loss kernel. Mixed layers contribute in proportion to their optical depth.
Scattering orders extend until the omitted Poisson probability is below
1e-10; the old fixed ``feature.plurality`` no longer limits camera spectra.
Zero-loss fraction is ``exp(-tau)``. Detector channels integrate probabilities,
so coarse dispersion does not lose the ZLP. Exposure/current scale counts;
energy binning sums them. Moving the energy window does not renormalize its
signal. Counts beyond the captured energy range are genuinely absent.

This remains a phenomenological model: Gaussian plasmons and smooth synthetic
core edges, default lambda 100 nm and core-event fraction 0.03. It does not
calculate material cross sections, ELNES, elastic/aperture losses, energy- or
voltage-dependent mean free paths, channeling or a finite convergent probe.
A mesh carries no chemistry. ``STL_EELS_EDGES``, ``STL_EELS_PLASMON_EV`` and
``STL_EELS_MEAN_FREE_PATH_NM`` explicitly assign one uniform synthetic Ni-like
material; change them in ``SimulationSettings.py`` and restart uSim. The
sphere uses the same default Ni-like parameters. These defaults are suitable
for testing geometric thickness trends, not quantitative material analysis.

EELS shot noise samples actual electrons per channel using camera gain. The
simulator frame includes ``eels_simulation`` metadata with local thickness,
optical depth, ideal zero-loss fraction and detector-window fraction. Device
record/sequence properties retain this snapshot; the installed Nion live
bridge may discard custom frame metadata. Spectrum images must not interpret
a single final-frame snapshot as a spatial thickness map.

Select **Spherical Particle**, center the stage at zero, set scan center zero
and FoV about 160 nm, then move the probe from the particle center to its rim.
For radius 50 nm and lambda 100 nm, center thickness is 100 nm with zero-loss
fraction 0.3679; at radius 40 nm these become 60 nm and 0.5488. Outside radius
50 nm the spectrum is vacuum ZLP. Restart Nion Swift to reload this code.

Run the physical model checks in nionswift-dev::

    python -m unittest nionswift_plugin.usim.test.EELSModel_test -v

Generate deterministic spectra and geometry CSV/NPZ for inspection::

    python tools/preview_eels_thickness.py --data-only

With Matplotlib installed, render the same data (this step can use a separate
plotting environment without Nion; activate that environment first so its
native DLL search paths are available)::

    python tools/preview_eels_thickness.py --render-only

Outputs are local and ignored under ``tools/eels_thickness_results/``.


Sample-specific initial views
-----------------------------

On startup and each specimen change, the ten-particle **STL Depth Sample** uses
stage x=1222 nm, y=279 nm and FoV 10000 nm. Every other sample starts at stage
(0, 0) with FoV 200 nm. Fast/Slow/Record profiles and scan center are updated
on selection, while pixel sizes and dwell times remain as configured.
Re-selecting the already active sample does not reset a manually adjusted view.
