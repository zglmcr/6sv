# py6s_only Code Explanation

This folder now keeps one current workflow: ljn MODIS/AERONET correction matched
by `oc_id`. The production entrypoint is:

```text
Code/py6s_only/ljn_ocid_py6s_correction.py
```

## Structure

```text
py6s_only/
  ljn_ocid_py6s_correction.py
  modules/
    common.py
    aeronet.py
    modis.py
    sixs_utils.py
  tools/
    audit_surface_reflectance_output.py
    aeronet_web_download.py
  docs/
    ljn_ocid_py6s_correction_notes.py
  outputs/
```

## Module Roles

- `modules/aeronet.py`
  - Defines the only AERONET record class: `LjnAeronetRecord`.
  - Reads `lwn_with_aod_inv15_ocid.csv`.
  - Extracts AOD, INV aerosol parameters, gas columns, and OC products.

- `modules/modis.py`
  - Defines `ModisSample`.
  - Reads one MODIS CSV row into geometry and TOA reflectance fields.

- `modules/common.py`
  - Provides generic parsing, wavelength matching, and interpolation helpers.

- `modules/sixs_utils.py`
  - Configures Py6S geometry, altitude, gas profile, wavelength, and ground.
  - Converts ozone from Dobson units to `cm-atm`.
  - Provides output formatting and Lambertian reflectance inversion.

## Main Workflow

1. Load AERONET records into an `oc_id -> LjnAeronetRecord` index.
2. Stream MODIS rows from `modis_l1b_result.csv`.
3. Match each MODIS row to AERONET by exact `oc_id`.
4. For each requested MODIS band:
   - estimate band AOD from the AERONET spectrum,
   - build an aerosol model from INV data when possible,
   - run Py6S for total path scattering,
   - run Py6S again for Rayleigh-only scattering,
   - write TOA, atmospheric terms, direct subtraction reflectance, Lambertian
     reflectance, and original OC products.

## Important Output Columns

- `rho_toa`: original MODIS TOA reflectance.
- `radiance_toa`: original MODIS radiance.
- `oc_rho`: original AERONET-OC `Rho[...]`.
- `oc_wavelength_um`: actual AERONET-OC wavelength matched to the MODIS band.
- `rho_path_total`: Py6S total atmospheric path reflectance.
- `rho_rayleigh`: Py6S Rayleigh-only path reflectance.
- `rho_aerosol`: `rho_path_total - rho_rayleigh`.
- `rho_toa_minus_atmosphere`: `rho_toa - rho_path_total`.
- `rho_surface_lambertian`: Lambertian inversion using Py6S transmittance and
  spherical albedo.

