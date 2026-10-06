# Public ultrasound datasets for the first validation stage

## 1. Recommended: Dryad 10.5061/dryad.47d7wm3qg

Data from: Monitoring microvascular changes over time with a repositionable 3D ultrasonic capacitive micromachined row-column sensor.

The dataset page reports:

- open data;
- 3-D beamformed ultrasound volumes;
- RCA/CMUT acquisition;
- repeated time frames;
- direct relevance to 3-D power Doppler / microvascular imaging;
- a supplied MATLAB script that applies SVD clutter filtering;
- PW3232: 37 x 64 x 64 x 400 x 2;
- PW1616: 37 x 64 x 64 x 800 x 2;
- SA64/HSA64: 37 x 64 x 64 x 400 x 1;
- voxel size 0.15 mm and frame rate 46.875/93.75 Hz.

Source:
https://doi.org/10.5061/dryad.47d7wm3qg

The dataset is the best first target because it is already volumetric and repeated in time, while the supplied processing explicitly uses SVD to obtain power-Doppler volumes.

## 2. Strong second source: Nature Methods 4D functional ultrasound

4D functional ultrasound imaging of whole-brain activity in rodents.

The associated publication provides example RF/IQ data and code for 3-D beamforming and clutter filtering. The data include IQ parts for 3-D clutter-filter processing and volumetric Doppler analysis.

Source:
https://www.nature.com/articles/s41592-019-0572-y

This is useful for a more complex 3-D time-series setting, but the data-access workflow is less convenient than Dryad.

## 3. Rat-brain ULM benchmark — 2-D control

In vivo rat brain for Ultrasound Localization Microscopy: raw and beamformed data.

Zenodo 10.5281/zenodo.7883227 provides raw RF and beamformed IQ data for 2-D rat-brain ULM.

Source:
https://doi.org/10.5281/zenodo.7883227

Use this later to test whether the sampling principle also helps conventional 2-D ultrafast ultrasound.

## 4. ULMShare — large external validation set

ULMShare provides a large public ULM library with raw ultrasonic data and multiple mice/procedures. The current public repository describes 99 acquisitions from 61 mice and the full raw archive.

Source:
https://doi.org/10.20383/103.01550
Repository:
https://github.com/provostultrasoundlab/ulmshare

The complete archive is tens of terabytes, so it is not the first smoke-test target.

## 5. EPFL Ultrafast Ultrasound Dataset — 2-D generalization

The EPFL dataset contains more than 20,000 in-vivo ultrafast acquisitions from nine volunteers plus phantom data, with acquisition settings and downloads covering abdomen, breast, carotid, limbs and other regions.

Source:
https://www.epfl.ch/labs/lts5/research/us/epfl-ultrafast-ultrasound-datasets/

It is not the primary 3-D Tetris target but is valuable later for testing generalization beyond brain/RCA data.

## Suggested validation order

1. Dryad RCA: SVD -> log -> random VALID.
2. Dryad RCA: SVD -> log -> SVD-guided VALID.
3. Nature Methods 4-D fUS for external volumetric validation.
4. ULMShare / Heiles Zenodo for broader external validation.
5. EPFL ultrafast data for non-brain and 2-D generalization.

Dataset licensing must be checked and cited separately. Do not assume a dataset license follows the code license.
