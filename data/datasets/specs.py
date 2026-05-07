"""Benchmark dataset catalogue.

Register benchmark-scope datasets here. Additional datasets should be added
explicitly with matching window defaults, documentation, and validation tests.
"""

from __future__ import annotations

from .base import DatasetRegistry, DatasetSpec


PENMANSHIEL_WT08_CHANNELS = (
    "Wind speed (m/s)",
    "Wind speed, Standard deviation (m/s)",
    "Wind speed, Minimum (m/s)",
    "Wind speed, Maximum (m/s)",
    "Long Term Wind (m/s)",
    "Wind speed Sensor 1 (m/s)",
    "Wind speed Sensor 1, Standard deviation (m/s)",
    "Wind speed Sensor 1, Minimum (m/s)",
    "Wind speed Sensor 1, Maximum (m/s)",
    "Wind speed Sensor 2 (m/s)",
    "Wind speed Sensor 2, Standard deviation (m/s)",
    "Wind speed Sensor 2, Minimum (m/s)",
    "Wind speed Sensor 2, Maximum (m/s)",
    "Wind direction (°)",
    "Nacelle position (°)",
    "Wind direction, Standard deviation (°)",
    "Wind direction, Minimum (°)",
    "Wind direction, Maximum (°)",
    "Nacelle position, Standard deviation (°)",
    "Nacelle position, Minimum (°)",
    "Nacelle position, Maximum (°)",
    "Power (kW)",
    "Power factor (cosphi)",
    "Power factor (cosphi), Max",
    "Power factor (cosphi), Min",
    "Power factor (cosphi), Standard deviation",
    "Reactive power (kvar)",
    "Reactive power, Max (kvar)",
    "Reactive power, Min (kvar)",
    "Reactive power, Standard deviation (kvar)",
    "Stator temperature 1 (°C)",
    "Generator bearing rear temperature (°C)",
    "Generator bearing front temperature (°C)",
    "Temp. top box (°C)",
    "Hub temperature (°C)",
    "Ambient temperature (converter) (°C)",
    "Rotor bearing temp (°C)",
    "Temperature motor axis 1 (°C)",
    "Temperature motor axis 2 (°C)",
    "Temperature motor axis 3 (°C)",
    "CPU temperature (°C)",
    "Generator bearing front temperature, Max (°C)",
    "Generator bearing front temperature, Min (°C)",
    "Generator bearing rear temperature, Max (°C)",
    "Generator bearing rear temperature, Min (°C)",
    "Grid voltage (V)",
    "Grid voltage, Max (V)",
    "Grid voltage, Min (V)",
    "Grid voltage, Standard deviation (V)",
    "Grid current (A)",
    "Motor current axis 1 (A)",
    "Motor current axis 2 (A)",
    "Motor current axis 3 (A)",
    "Rotor speed (RPM)",
    "Generator RPM (RPM)",
    "Generator RPM, Max (RPM)",
    "Generator RPM, Min (RPM)",
    "Generator RPM, Standard deviation (RPM)",
    "Rotor speed, Max (RPM)",
    "Rotor speed, Min (RPM)",
    "Rotor speed, Standard deviation (RPM)",
    "Gear oil inlet pressure (bar)",
    "Gear oil pump pressure (bar)",
    "Grid frequency (Hz)",
    "Drive train acceleration (mm/ss)",
)

BEIJING_AIR_TIANTAN_CONTINUOUS_CHANNELS = (
    "PM2.5",
    "PM10",
    "SO2",
    "NO2",
    "CO",
    "O3",
    "TEMP",
    "PRES",
    "DEWP",
    "RAIN",
    "WSPM",
)
BEIJING_AIR_TIANTAN_DISCRETE_CHANNELS = ("wd",)


DATASET_REGISTRY = DatasetRegistry(
    DatasetSpec(
        key="Penmanshiel_Hourly_WT08",
        path="penmanshiel_hourly_wt08.parquet",
        split_mode="temporal",
        input_channels=PENMANSHIEL_WT08_CHANNELS,
        channel_groups={"power": ("Power (kW)",)},
        default_target="power",
        description=(
            "Penmanshiel wind farm SCADA (WT08, hourly). "
            "Created from the Zenodo v3 2016-2022 WT08 SCADA subset only. "
            "Window 2016-08-18 15:00 to 2019-08-01 09:00 selected as the longest WT08 segment after splitting target gaps longer than one day and removing channels whose kept-segment gaps exceed 21 days. "
            "Hourly mean aggregation of 10-minute period averages on completed hourly bins. "
            "Power (kW) forecast target."
        ),
        continuous_channels=PENMANSHIEL_WT08_CHANNELS,
        discrete_channels=(),
    ),
    DatasetSpec(
        key="ETTh1",
        path="ETTh1.csv",
        split_mode="temporal",
        input_channels=(
            "HUFL",
            "HULL",
            "MUFL",
            "MULL",
            "LUFL",
            "LULL",
            "OT",
        ),
        channel_groups={
            "ot": ("OT",),
            "all": ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"),
        },
        default_target="all",
        description="ETTh1 multivariate time-series (7 channels).",
        continuous_channels=("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"),
        discrete_channels=(),
    ),
    DatasetSpec(
        key="BeijingAir_Tiantan",
        path="beijing_air_tiantan.parquet",
        split_mode="temporal",
        input_channels=(
            BEIJING_AIR_TIANTAN_CONTINUOUS_CHANNELS
            + BEIJING_AIR_TIANTAN_DISCRETE_CHANNELS
        ),
        channel_groups={"pm25": ("PM2.5",)},
        default_target="pm25",
        description=(
            "Beijing Multi-Site Air Quality (PRSA, Tiantan only, hourly, 2014-2017 trimmed). "
            "Created from raw Tiantan station data. "
            "Window 2014-06-03 10:00 to 2017-02-28 23:00 selected by allowing exact 24-hour PM2.5 gaps but splitting non-target feature gaps longer than 3 days while preserving the full local variable set. "
            "PM2.5 forecast target. "
            "Wind direction (wd) is integer-coded (0-15, clockwise from N) and treated as discrete."
        ),
        continuous_channels=BEIJING_AIR_TIANTAN_CONTINUOUS_CHANNELS,
        discrete_channels=BEIJING_AIR_TIANTAN_DISCRETE_CHANNELS,
    ),
    (lambda channels: DatasetSpec(
        key="traffic",
        path="traffic.parquet",
        split_mode="temporal",
        input_channels=channels,
        channel_groups={"all": channels},
        default_target="all",
        description=(
            "Traffic occupancy benchmark (862 hourly sensor series). "
            "PeMS-derived San Francisco Bay Area freeway loop-detector occupancy data, "
            "aggregated to hourly resolution. "
            "Each channel is one fixed detector measuring occupancy "
            "(fraction of the interval covered by vehicles). "
            "Standard benchmark derivative used by Autoformer/TimesNet."
        ),
        continuous_channels=channels,
        discrete_channels=(),
    ))(tuple(str(i) for i in range(861)) + ("OT",)),
)

__all__ = ["DATASET_REGISTRY", "DatasetSpec"]
