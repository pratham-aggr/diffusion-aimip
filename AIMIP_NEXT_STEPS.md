# AIMIP-1 Submission: Next Steps After Training

## Overview
This guide covers the steps to generate AIMIP-1 compliant submissions after completing training with `era5_sst_seaice_emulation_edm_aimip_tuned_v2`.

## 1. Training Completion Checklist

- [ ] Training completed successfully
- [ ] Best checkpoint saved (monitored by `val/avg/rmse`)
- [ ] Model weights available
- [ ] Validation metrics logged to wandb

## 2. Generate Inference Simulations

### 2.1 Run 5-Ensemble AMIP Simulations

Generate 5 ensemble members for the standard AMIP case:

```bash
# Run inference for ensemble members r1i1p1f1 through r5i1p1f1
# Start: 00 UTC 1 Oct 1978
# End: 00 UTC 1 Jan 2025 (46.25 years)
# Spin-up: First 3 months (Oct-Dec 1978)

python run_aimip_inference.py \
    --experiment era5_sst_seaice_emulation_edm_aimip_tuned_v2 \
    --checkpoint_path <path_to_best_checkpoint> \
    --ensemble_members 5 \
    --start_date "1978-10-01" \
    --end_date "2025-01-01" \
    --output_dir ./aimip_submissions/amip_standard
```

### 2.2 Optional: AMIP+2K and AMIP+4K Simulations

```bash
# AMIP+2K (SST uniformly increased by 2K)
python run_aimip_inference.py \
    --experiment era5_sst_seaice_emulation_edm_aimip_tuned_v2 \
    --checkpoint_path <path_to_best_checkpoint> \
    --sst_offset 2.0 \
    --ensemble_members 5 \
    --output_dir ./aimip_submissions/amip_plus2k

# AMIP+4K (SST uniformly increased by 4K)
python run_aimip_inference.py \
    --experiment era5_sst_seaice_emulation_edm_aimip_tuned_v2 \
    --checkpoint_path <path_to_best_checkpoint> \
    --sst_offset 4.0 \
    --ensemble_members 5 \
    --output_dir ./aimip_submissions/amip_plus4k
```

## 3. Post-Processing Required Fields

### 3.1 Surface Temperature (ts)

**Issue**: `surface_temperature` not available in ERA5 dataset.

**Solution**: Compute during post-processing:
- Over ocean: Use `sea_surface_temperature`
- Over land: Use `2m_temperature` as proxy
- Over sea ice: Use `2m_temperature` as proxy

```python
# Pseudocode for post-processing
surface_temp = np.where(
    land_sea_mask == 0,  # Ocean
    sea_surface_temperature,
    2m_temperature  # Land/sea ice
)
```

### 3.2 Precipitation Rate Conversion

**Issue**: `total_precipitation_6hr` is 6-hourly accumulated, needs conversion to rate.

**Solution**: Divide by 21600 seconds (6 hours)

```python
precipitation_rate = total_precipitation_6hr / 21600.0  # Convert to kg/(m² s)
```

### 3.3 2m Dewpoint Temperature (tdas) - Optional

**Status**: Not currently included. If required:
- Verify ERA5 variable name for 2m dewpoint
- Or compute from `2m_temperature` and `specific_humidity_1000` using ECMWF formulas
- Reference: https://prod.ecmwf-forum-prod.compute.cci2.ecmwf.int/t/how-to-calculate-hus-at-2m-huss/1254

## 4. CMORization

Convert outputs to CMIP-compliant format:

```bash
python src/utilities/aimip_cmorize.py \
    --input_dir ./aimip_submissions/amip_standard \
    --output_dir ./aimip_submissions/cmorized \
    --model_name "EDM-AIMIP" \
    --experiment_id "aimip" \
    --table_id "Amon" \
    --start_date "197810" \
    --end_date "202412"
```

### CMOR Requirements:
- **File naming**: `CFfieldname_Amon_MMM_aimip_rXiXpXfX_gn_197810-202412.nc`
- **Grid**: Native grid (`gn` label)
- **Precision**: Single precision (float32)
- **Time**: Monthly means (Oct 1978 - Dec 2024)
- **Units**: CF-compliant units (see `src/utilities/aimip_cmor_mapping.py`)

## 5. Output Files Structure

```
aimip_submissions/
├── amip_standard/
│   ├── ta_Amon_EDM-AIMIP_aimip_r1i1p1f1_gn_197810-202412.nc
│   ├── hus_Amon_EDM-AIMIP_aimip_r1i1p1f1_gn_197810-202412.nc
│   ├── ua_Amon_EDM-AIMIP_aimip_r1i1p1f1_gn_197810-202412.nc
│   ├── va_Amon_EDM-AIMIP_aimip_r1i1p1f1_gn_197810-202412.nc
│   ├── ps_Amon_EDM-AIMIP_aimip_r1i1p1f1_gn_197810-202412.nc
│   ├── psl_Amon_EDM-AIMIP_aimip_r1i1p1f1_gn_197810-202412.nc
│   ├── ts_Amon_EDM-AIMIP_aimip_r1i1p1f1_gn_197810-202412.nc  # Post-processed
│   ├── tas_Amon_EDM-AIMIP_aimip_r1i1p1f1_gn_197810-202412.nc
│   ├── uas_Amon_EDM-AIMIP_aimip_r1i1p1f1_gn_197810-202412.nc
│   ├── vas_Amon_EDM-AIMIP_aimip_r1i1p1f1_gn_197810-202412.nc
│   ├── pr_Amon_EDM-AIMIP_aimip_r1i1p1f1_gn_197810-202412.nc  # Converted to rate
│   ├── zg_Amon_EDM-AIMIP_aimip_r1i1p1f1_gn_197810-202412.nc
│   └── ... (repeat for r2i1p1f1 through r5i1p1f1)
├── amip_plus2k/  # Optional
└── amip_plus4k/  # Optional
```

## 6. Required Output Fields Summary

### 3D Fields (7 pressure levels: 1000, 850, 700, 500, 250, 100, 50 hPa)
- ✅ `ta` - Temperature [K]
- ✅ `hus` - Specific humidity [kg/kg]
- ✅ `ua` - Eastward wind [m/s]
- ✅ `va` - Northward wind [m/s]

### Surface Fields
- ✅ `ps` - Surface pressure [Pa]
- ✅ `psl` - Sea-level pressure [Pa]
- ⚠️ `ts` - Surface temperature [K] - **Needs post-processing**
- ✅ `tas` - 2m air temperature [K]
- ⚠️ `tdas` - 2m dewpoint temperature [K] - **Optional, not included**
- ✅ `uas` - 10m eastward wind [m/s]
- ✅ `vas` - 10m northward wind [m/s]
- ⚠️ `pr` - Precipitation rate [kg/(m² s)] - **Needs conversion**

### Geopotential
- ✅ `zg` - 500 hPa geopotential height [m]

## 7. Daily Outputs (Optional)

For daily data (Oct 1978 - Dec 1979 and full year 2024):

```bash
python run_aimip_inference.py \
    --experiment era5_sst_seaice_emulation_edm_aimip_tuned_v2 \
    --checkpoint_path <path_to_best_checkpoint> \
    --save_daily \
    --daily_periods "1978-10-01,1979-12-31" "2024-01-01,2024-12-31" \
    --output_dir ./aimip_submissions/daily_outputs
```

**File naming**: `CFfieldname_day_MMM_aimip_rXiXpXfX_gn_19781001-19791231.nc`

## 8. Validation Checklist

Before submission, verify:

- [ ] All 5 ensemble members generated
- [ ] Monthly outputs cover full period (Oct 1978 - Dec 2024)
- [ ] All required fields present
- [ ] `surface_temperature` computed correctly
- [ ] `precipitation_rate` converted to rate (not accumulated)
- [ ] CMIP naming convention followed
- [ ] Units are CF-compliant
- [ ] Grid documented (native grid with `gn` label)
- [ ] No NaN values in outputs
- [ ] File sizes reasonable (~5 GB per simulation for monthly data)

## 9. Submission

Contact: Chris Bretherton (christopherb@allenai.org)

**Deadlines**:
- Initial submission: Nov 30, 2025
- Warmed-SST cases: Dec 31, 2025

**Storage**: DKRZ will store submissions using EERIE cloud-based distribution

## 10. Documentation Requirements

Submit with your model:
- [ ] Model weights (for reproducibility)
- [ ] Training approach documentation
- [ ] Hyperparameter settings
- [ ] Model architecture description
- [ ] Any custom post-processing scripts

## Quick Reference

**Training Config**: `src/configs/experiment/era5_sst_seaice_emulation_edm_aimip_tuned_v2.yaml`
**DataModule Config**: `src/configs/datamodule/era5_sst_seaice_aimip.yaml`
**CMOR Mapping**: `src/utilities/aimip_cmor_mapping.py`
**AIMIP Spec**: `AIMIP.md`

## Notes

- Surface temperature (`ts`) must be computed during post-processing
- Precipitation must be converted from accumulated to rate
- 2m dewpoint temperature (`tdas`) is optional but listed in AIMIP.md
- All outputs should be on native grid (`gn` label)
- Use single precision (float32) for storage efficiency
