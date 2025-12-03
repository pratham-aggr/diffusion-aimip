"""
CMIP Variable Name Mapping for AIMIP-1 Submission

Maps internal variable names to CMIP CF-compliant names and units.
Based on CMIP6/CMIP7 specifications and AIMIP-1 requirements.
"""

from typing import Dict, Optional

# CMIP variable name mapping: internal_name -> CMIP_name
CMIP_VARIABLE_MAPPING: Dict[str, str] = {
    # 3D Fields at pressure levels
    "temperature_1000": "ta",  # Temperature at 1000 hPa
    "temperature_850": "ta",
    "temperature_700": "ta",
    "temperature_500": "ta",
    "temperature_250": "ta",
    "temperature_100": "ta",
    "temperature_50": "ta",
    
    "specific_humidity_1000": "hus",  # Specific humidity at 1000 hPa
    "specific_humidity_850": "hus",
    "specific_humidity_700": "hus",
    "specific_humidity_500": "hus",
    "specific_humidity_250": "hus",
    "specific_humidity_100": "hus",
    "specific_humidity_50": "hus",
    
    "u_component_of_wind_1000": "ua",  # Eastward wind at 1000 hPa
    "u_component_of_wind_850": "ua",
    "u_component_of_wind_700": "ua",
    "u_component_of_wind_500": "ua",
    "u_component_of_wind_250": "ua",
    "u_component_of_wind_100": "ua",
    "u_component_of_wind_50": "ua",
    
    "v_component_of_wind_1000": "va",  # Northward wind at 1000 hPa
    "v_component_of_wind_850": "va",
    "v_component_of_wind_700": "va",
    "v_component_of_wind_500": "va",
    "v_component_of_wind_250": "va",
    "v_component_of_wind_100": "va",
    "v_component_of_wind_50": "va",
    
    "geopotential_500": "zg",  # Geopotential height at 500 hPa
    
    # Surface fields
    "surface_pressure": "ps",  # Surface pressure
    "mean_sea_level_pressure": "psl",  # Sea-level pressure
    "surface_temperature": "ts",  # Surface (skin) temperature
    "2m_temperature": "tas",  # 2m air temperature
    "10m_u_component_of_wind": "uas",  # 10m eastward wind
    "10m_v_component_of_wind": "vas",  # 10m northward wind
    "total_column_water_vapour": "prw",  # Precipitable water (optional)
    "total_precipitation_6hr": "pr",  # Precipitation rate (6-hourly accumulated, needs conversion to rate)
    # "2m_dewpoint_temperature": "tdas",  # 2m dewpoint temperature (verify variable name in ERA5)
}

# CMIP units mapping
CMIP_UNITS_MAPPING: Dict[str, str] = {
    "ta": "K",  # Temperature
    "hus": "kg kg-1",  # Specific humidity
    "ua": "m s-1",  # Eastward wind
    "va": "m s-1",  # Northward wind
    "zg": "m",  # Geopotential height
    "ps": "Pa",  # Surface pressure
    "psl": "Pa",  # Sea-level pressure
    "ts": "K",  # Surface temperature
    "tas": "K",  # 2m air temperature
    "uas": "m s-1",  # 10m eastward wind
    "vas": "m s-1",  # 10m northward wind
    "prw": "kg m-2",  # Precipitable water
    "pr": "kg m-2 s-1",  # Precipitation rate
}

# CMIP long names
CMIP_LONG_NAMES: Dict[str, str] = {
    "ta": "Air Temperature",
    "hus": "Specific Humidity",
    "ua": "Eastward Wind",
    "va": "Northward Wind",
    "zg": "Geopotential Height",
    "ps": "Surface Air Pressure",
    "psl": "Sea Level Pressure",
    "ts": "Surface Temperature",
    "tas": "Near-Surface Air Temperature",
    "uas": "Eastward Near-Surface Wind",
    "vas": "Northward Near-Surface Wind",
    "prw": "Precipitable Water",
    "pr": "Precipitation",
}

# AIMIP-1 required pressure levels (in hPa, converted to Pa for CMIP)
AIMIP_PRESSURE_LEVELS_HPA = [1000, 850, 700, 500, 250, 100, 50]
AIMIP_PRESSURE_LEVELS_PA = [p * 100 for p in AIMIP_PRESSURE_LEVELS_HPA]  # Convert to Pa


def get_cmip_name(internal_name: str) -> Optional[str]:
    """Get CMIP variable name from internal variable name."""
    # Extract base name and level if present
    if "_" in internal_name:
        parts = internal_name.split("_")
        # Check if last part is a number (pressure level)
        if parts[-1].isdigit():
            base_name = "_".join(parts[:-1])
            if base_name in CMIP_VARIABLE_MAPPING:
                return CMIP_VARIABLE_MAPPING[base_name]
    # Direct mapping
    return CMIP_VARIABLE_MAPPING.get(internal_name)


def get_cmip_units(cmip_name: str) -> Optional[str]:
    """Get CMIP units for a variable."""
    return CMIP_UNITS_MAPPING.get(cmip_name)


def get_cmip_long_name(cmip_name: str) -> Optional[str]:
    """Get CMIP long name for a variable."""
    return CMIP_LONG_NAMES.get(cmip_name)


def extract_pressure_level(internal_name: str) -> Optional[int]:
    """Extract pressure level in hPa from variable name."""
    if "_" in internal_name:
        parts = internal_name.split("_")
        if parts[-1].isdigit():
            return int(parts[-1])
    return None


def get_aimip_output_variables() -> Dict[str, list]:
    """
    Get AIMIP-1 required output variables organized by type.
    
    Returns:
        dict with keys: '3d_fields', 'surface_fields', 'pressure_levels'
    """
    return {
        "3d_fields": ["ta", "hus", "ua", "va"],
        "surface_fields": ["ps", "psl", "ts", "tas", "uas", "vas", "pr"],
        "pressure_levels": AIMIP_PRESSURE_LEVELS_PA,  # In Pa for CMIP
        "optional_fields": ["prw", "pr"],  # Precipitable water, precipitation
        "geopotential": ["zg"],  # At 500 hPa
    }
