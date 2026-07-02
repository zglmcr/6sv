# Py6S-only Atmospheric Correction

This folder keeps only the Py6S atmospheric-correction step.

It assumes the input MODIS TOA sample CSV already exists.  The CSV must already contain matched sample time, location, solar/view geometry, and TOA reflectance bands.  This folder does not read MODIS L1B HDF files and does not run OCSSW.

## Files

- `ljn_ocid_py6s_correction.py`: main ljn `oc_id` matched MODIS/AERONET Py6S correction entrypoint.
- `modules/common.py`: shared parsing, wavelength matching, and interpolation helpers.
- `modules/aeronet.py`: AERONET ljn table reader, `oc_id` index builder, AOD/INV/OC column extraction.
- `modules/modis.py`: MODIS ljn row parser and TOA reflectance sample builder.
- `modules/sixs_utils.py`: small Py6S setup, ozone conversion, formatting, and Lambertian inversion utilities.
- `tools/audit_surface_reflectance_output.py`: output consistency audit.
- `tools/aeronet_web_download.py`: optional AERONET web download helper.
- `docs/ljn_ocid_py6s_correction_notes.py`: concise reading notes for the ljn main script.
- `run_example.ps1`: runs the bundled example input through correction and audit.
- `examples/`: small example input tables.
- `outputs/`: generated results.

## Python Environment

Use this existing interpreter in PyCharm:

```text
C:\Users\Administrator\Desktop\一致性\Code\Py6SV\envs\py6s\python.exe
```

The 6S executable used by default is:

```text
C:\Users\Administrator\Desktop\一致性\Code\Py6SV\envs\py6s\Library\bin\sixs.exe
```

Because the environment folder is named `Py6SV`, it no longer conflicts with the third-party Python package named `Py6S`.

## Run Small Check

From the project root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File Code\py6s_only\run_example.ps1
```

The small-check output is:

```text
Code\py6s_only\outputs\ljn_example_surface_reflectance.csv
```

Important output columns:

- `rho_toa`: input TOA reflectance.
- `rho_path_total`: Py6S total atmospheric path reflectance.
- `rho_rayleigh`: Rayleigh-only path reflectance.
- `rho_aerosol`: `rho_path_total - rho_rayleigh`.
- `rho_surface_minus_path`: direct subtraction, `rho_toa - rho_path_total`.
- `rho_surface_lambertian`: Lambertian surface-reflectance inversion from 6S terms.
- `aerosol_source`: actual aerosol-model input source used by the ljn workflow.

## LJN oc_id Correction

For `Data\ljn`, use the dedicated script:

```powershell
Code\Py6SV\envs\py6s\python.exe Code\py6s_only\ljn_ocid_py6s_correction.py `
  --modis-csv Data\ljn\modis_l1b_result.csv `
  --aeronet-csv Data\ljn\lwn_with_aod_inv15_ocid.csv
```

It writes only these default result files:

```text
Code\py6s_only\outputs\ljn_ocid_surface_reflectance.csv
Code\py6s_only\outputs\ljn_ocid_surface_reflectance_summary.json
```

Key columns in the CSV:

- `oc_rho`: original AERONET-OC `Rho[...]` value for the closest available OC wavelength.
- `oc_wavelength_um`: actual OC wavelength used for `oc_rho`, `oc_lw`, and `oc_lwn`.
- `rho_toa`: original MODIS TOA reflectance.
- `radiance_toa`: original MODIS L1B radiance for the same band.
- `rho_path_total`: total atmospheric path reflectance from Py6S.
- `rho_rayleigh`: Rayleigh-only path reflectance from Py6S.
- `rho_aerosol`: aerosol path reflectance, `rho_path_total - rho_rayleigh`.
- `rho_toa_minus_atmosphere`: direct result, `rho_toa - rho_path_total`.
- `rho_surface_lambertian`: Lambertian inversion using transmittance and spherical albedo.
