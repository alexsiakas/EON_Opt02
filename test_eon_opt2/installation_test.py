
import eon_opt02

lightcurve = eon_opt02.LightCurve('prototype_input.kvn')
lightcurve.analyse(
    period_max=2.0, 
    period_min=0.5, 
    period_step=0.01, 
    fap_limit=0.001, 
    long_period_peak_ratio=0.9,
    cleaning_max_power_ratio=0.2, 
    cleaning_alliase_proximity_ratio=0.2, 
    pdm_bins=20,
    half_window=10, 
    poly_deg=1, 
    limit_to_single_winow=5, 
    single_window_poly_deg=2,
    export='.')
