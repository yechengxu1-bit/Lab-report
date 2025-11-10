import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed to register 3D projection)
from datetime import datetime, timedelta
import matplotlib.dates as mdates

# =========================
# 0) Core math helpers
# =========================
def ecef_to_lla(x, y, z):
    """Convert ECEF coordinates (meters) to geodetic coordinates (lat [deg], lon [deg], h [m])
    using the WGS-84 ellipsoid.
    Input may be scalars or numpy arrays; returns a tuple (lat_deg, lon_deg, h_m) where lat/lon are
    degrees and h is in meters.
    """
    # WGS-84 constants
    a = 6378137.0
    e2 = 6.69437999014e-3

    # Compute longitude directly
    lon = np.arctan2(y, x)

    # Intermediate values for iterative latitude solution
    p = np.sqrt(x**2 + y**2)
    # Initial guess for latitude (Bowring's method-ish)
    lat = np.arctan2(z, p * (1 - e2))
    lat_prev = 0.0

    # Radius of curvature in the prime vertical
    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    # Height from ellipsoid
    h = p / np.cos(lat) - N

    # Iteratively refine latitude and height until change is negligible
    while np.max(np.abs(lat - lat_prev)) > 1e-12:
        lat_prev = lat
        N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
        h = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1 - e2 * N / (N + h)))

    # Return degrees for lat/lon and meters for height
    return np.degrees(lat), np.degrees(lon), h


def ecef_to_enu(x, y, z, x_ref, y_ref, z_ref):
    """Convert ECEF coordinates to local East-North-Up (ENU) coordinates relative
    to a reference ECEF point (x_ref, y_ref, z_ref).
    Returns: numpy array [E, N, U]
    Note: This function calls ecef_to_lla to obtain the reference lat/lon.
    """
    # Reference latitude/longitude in radians
    lat_ref, lon_ref, _ = ecef_to_lla(x_ref, y_ref, z_ref)
    lat_ref = np.radians(lat_ref); lon_ref = np.radians(lon_ref)

    # Vector from reference to point
    dx, dy, dz = x - x_ref, y - y_ref, z - z_ref

    # Rotation matrix from ECEF->ENU
    t = np.array([
        [-np.sin(lon_ref),                  np.cos(lon_ref),                  0],
        [-np.sin(lat_ref)*np.cos(lon_ref), -np.sin(lat_ref)*np.sin(lon_ref),  np.cos(lat_ref)],
        [ np.cos(lat_ref)*np.cos(lon_ref),  np.cos(lat_ref)*np.sin(lon_ref),  np.sin(lat_ref)]
    ])

    # Apply rotation
    return t @ np.array([dx, dy, dz])


def deg_to_dms(value, is_lat):
    """Format decimal degrees as degrees-minutes-seconds string with hemisphere.
    is_lat: True if latitude (N/S), False if longitude (E/W)
    Example: 51.1789 -> "51°10′44.0″N"
    """
    deg = int(abs(value))
    minutes = int((abs(value) - deg) * 60)
    seconds = (abs(value) - deg - minutes/60) * 3600
    hemi = ('N' if value >= 0 else 'S') if is_lat else ('E' if value >= 0 else 'W')
    return f"{deg}°{minutes:02d}′{seconds:04.1f}″{hemi}"


def moving_average(x, k=9):
    """Symmetric moving average with odd window length k.
    Pads using edge values so output length equals input length.
    k is forced to an odd integer >= 1.
    """
    k = max(1, int(k) | 1)  # ensure odd integer
    pad = k // 2
    xpad = np.pad(np.asarray(x), (pad, pad), mode='edge')
    return np.convolve(xpad, np.ones(k)/k, mode='valid')


# =========================
# 1) Data loading (GT + SPP)
# =========================
def load_ground_truth(nav_hpposecef_csv):
    """Load NAV-HPPOSECEF ground truth CSV.
    Expected CSV layout: numeric columns with at least 8 columns; column indices used:
      - iTOW: column 2
      - X_cm, Y_cm, Z_cm: columns 3,4,5 (centimeters)
      - invalid flag: column 6 (0 = valid)
    Optionally, fine corrections in columns 8-10 (0.1 mm -> m) are applied if present.
    Returns a dict with ECEF in meters, LLA arrays, and iTOW.
    """
    data = np.genfromtxt(nav_hpposecef_csv, delimiter=',', skip_header=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 8:
        raise ValueError("NAV-HPPOSECEF.csv must have ≥8 numeric columns.")

    iTOW = data[:, 2]
    x_cm, y_cm, z_cm = data[:, 3], data[:, 4], data[:, 5]
    invalid = data[:, 6]

    # Keep only valid rows and finite values
    valid = (invalid == 0) & np.isfinite(x_cm) & np.isfinite(y_cm) & np.isfinite(z_cm)
    x, y, z = x_cm[valid]*0.01, y_cm[valid]*0.01, z_cm[valid]*0.01  # convert cm->m
    iTOW = iTOW[valid]

    # Apply optional high-precision offsets (0.1 mm -> m) if present and finite
    if data.shape[1] >= 11 and np.all(np.isfinite(data[valid, 8:11])):
        x += data[valid, 8]*1e-4
        y += data[valid, 9]*1e-4
        z += data[valid, 10]*1e-4

    # Convert ECEF -> LLA for each point
    lat, lon, alt = [], [], []
    for xx, yy, zz in zip(x, y, z):
        la, lo, al = ecef_to_lla(xx, yy, zz)
        lat.append(la); lon.append(lo); alt.append(al)

    return dict(ecef_x=x, ecef_y=y, ecef_z=z,
                lat=np.array(lat), lon=np.array(lon), alt=np.array(alt),
                itow=iTOW)


def load_estimated_solution(sat_positions_csv, pr_csv, clk_csv, ion_csv, trop_csv):
    """Perform Single Point Positioning (SPP) via iterative least-squares per epoch.
    Inputs are CSVs:
      - sat_positions_csv: per-satellite ECEF positions arranged as [sat, 3*epoch_cols]
      - pr_csv: pseudoranges (rows=sat, cols=epoch)
      - clk_csv: satellite clock biases (meters)
      - ion_csv: ionospheric delay corrections (meters)
      - trop_csv: tropospheric delay corrections (meters)
    Returns dict with estimated ECEF positions, lat/lon/alt arrays, and receiver clock bias (m).
    """
    # Load numeric arrays
    sat_pos = np.loadtxt(sat_positions_csv, delimiter=',')
    pr      = np.loadtxt(pr_csv, delimiter=',')
    clk     = np.loadtxt(clk_csv, delimiter=',')
    ion     = np.loadtxt(ion_csv, delimiter=',')
    trop    = np.loadtxt(trop_csv, delimiter=',')

    num_epochs = pr.shape[1]

    est_ecef, clk_biases = [], []
    # Initialize receiver state guess
    rx_pos = np.array([0.0, 0.0, 0.0])
    rx_clk_prev = 0.0

    def lsq(sats, rx0, pr_m, clk_m, ion_m, trop_m):
        """Inner least-squares solver for single epoch.
        sats: Nx3 satellite ECEF positions
        rx0: initial receiver ECEF guess
        pr_m, clk_m, ion_m, trop_m: per-satellite corrections (1D arrays)
        Returns: (rx_est, rx_clock_bias)
        """
        rx = rx0.copy()
        rx_clk = 0.0
        for _ in range(10):  # up to 10 iterations for convergence
            rho = np.linalg.norm(sats - rx, axis=1)  # geometric ranges
            prcorr = pr_m + clk_m - ion_m - trop_m  # corrected pseudoranges
            resid = prcorr - (rho + rx_clk)  # residuals (observed - predicted)
            # Design matrix: [-unit_vector, 1] per satellite
            G = np.zeros((len(sats), 4))
            for i in range(len(sats)):
                u = (sats[i] - rx) / rho[i]
                G[i,:3] = -u
                G[i,3] = 1.0
            # Solve normal equations (least-squares step)
            delta, *_ = np.linalg.lstsq(G, resid, rcond=None)
            rx += delta[:3]
            rx_clk += delta[3]
            if np.linalg.norm(delta[:3]) < 1e-4:  # convergence threshold (meters)
                break
        return rx, rx_clk

    # Process each epoch
    for k in range(num_epochs):
        # Extract per-epoch satellite positions (each sat row has 3 columns per epoch)
        sats_k = sat_pos[:, 3*k:3*(k+1)]
        # Determine satellites that have finite data for this epoch
        ok = np.isfinite(pr[:,k]) & np.isfinite(clk[:,k]) & np.isfinite(ion[:,k]) \
             & np.isfinite(trop[:,k]) & np.isfinite(sats_k).all(axis=1)

        # If fewer than 4 satellites available, cannot compute a fix: keep previous estimate
        if np.sum(ok) < 4:
            est_ecef.append(est_ecef[-1] if est_ecef else rx_pos.copy())
            clk_biases.append(clk_biases[-1] if clk_biases else rx_clk_prev)
            continue

        # Solve for this epoch using only the 'ok' satellites
        rx_pos, rx_clk = lsq(sats_k[ok,:], rx_pos, pr[ok,k], clk[ok,k], ion[ok,k], trop[ok,k])
        est_ecef.append(rx_pos.copy())
        clk_biases.append(rx_clk)

    # Convert accumulated estimates to arrays and compute LLA
    est_ecef = np.array(est_ecef)
    lat, lon, alt = [], [], []
    for p in est_ecef:
        la, lo, al = ecef_to_lla(*p)
        lat.append(la); lon.append(lo); alt.append(al)

    return dict(ecef=est_ecef, lat=np.array(lat), lon=np.array(lon),
                alt=np.array(alt), clk_bias_m=np.array(clk_biases))


# =========================
# 2) DOP computation & plot (time-based hh:mm:ss)
# =========================
def _los_unit_vectors(sat_ecef, rx_ecef):
    """Compute line-of-sight unit vectors from receiver to satellites.
    Returns:
      - u: Nx3 array of unit vectors (sat - rx) / range
      - ok: boolean mask for valid vectors
    """
    rho = sat_ecef - rx_ecef[None, :]
    dist = np.linalg.norm(rho, axis=1, keepdims=True)
    ok = (dist[:, 0] > 0) & np.all(np.isfinite(rho), axis=1)
    u = np.zeros_like(rho)
    u[ok] = rho[ok] / dist[ok]
    return u, ok


def _dop_from_geometry(u):
    """Compute DOP (GDOP, PDOP, HDOP, VDOP, TDOP) from LOS unit vectors.
    If fewer than 4 vectors or matrix singular, returns NaNs.
    """
    if u.shape[0] < 4:
        return dict(GDOP=np.nan, PDOP=np.nan, HDOP=np.nan, VDOP=np.nan, TDOP=np.nan)

    # H matrix: [-u | 1] per satellite (design matrix)
    H = np.hstack([-u, np.ones((u.shape[0], 1))])
    try:
        Q = np.linalg.inv(H.T @ H)  # covariance of LS solution (geometry only)
    except np.linalg.LinAlgError:
        # Singular geometry -> cannot compute DOP
        return dict(GDOP=np.nan, PDOP=np.nan, HDOP=np.nan, VDOP=np.nan, TDOP=np.nan)

    PDOP = np.sqrt(Q[0,0] + Q[1,1] + Q[2,2])
    HDOP = np.sqrt(Q[0,0] + Q[1,1])
    VDOP = np.sqrt(Q[2,2])
    TDOP = np.sqrt(Q[3,3])
    GDOP = np.sqrt(np.trace(Q))
    return dict(GDOP=GDOP, PDOP=PDOP, HDOP=HDOP, VDOP=VDOP, TDOP=TDOP)


def compute_dops_over_time(satellite_positions_csv, rx_ecef_series):
    """Compute DOP metrics for each epoch using satellite positions CSV and receiver ECEF series.
    Returns a dict of arrays for GDOP, PDOP, HDOP, VDOP, TDOP (same length as epochs processed).
    """
    sat_pos = np.loadtxt(satellite_positions_csv, delimiter=',')
    max_sats, total_cols = sat_pos.shape
    num_epochs = total_cols // 3
    # Limit to the length of the provided receiver estimates
    num = min(num_epochs, len(rx_ecef_series))

    dops = {k: np.full(num, np.nan) for k in ["GDOP","PDOP","HDOP","VDOP","TDOP"]}
    for k in range(num):
        block = sat_pos[:, 3*k:3*(k+1)]
        valid = np.all(np.isfinite(block), axis=1)
        sats_k = block[valid, :]
        # Compute LOS unit vectors with respect to receiver position at epoch k
        u, ok = _los_unit_vectors(sats_k, rx_ecef_series[k])
        res = _dop_from_geometry(u[ok])
        for key in dops:
            dops[key][k] = res[key]
    return dops


def plot_dops_time(dops, nav_hpposecef_csv, show_means=True, utc_offset_hours=0):
    """Plot DOP metrics over time using iTOW times from NAV-HPPOSECEF CSV.
    The times are built using today's UTC midnight + iTOW (ms). utc_offset_hours can shift display time.
    """
    nav = np.genfromtxt(nav_hpposecef_csv, delimiter=',', skip_header=1)
    iTOW_ms = nav[:, 2]

    # Build datetime list starting from today's UTC midnight (for hh:mm:ss axis)
    t0 = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    times = [t0 + timedelta(milliseconds=float(t)) + timedelta(hours=utc_offset_hours) for t in iTOW_ms]

    order = ["GDOP","PDOP","HDOP","VDOP","TDOP"]
    colors = {"GDOP":"k","PDOP":"tab:blue","HDOP":"tab:green","VDOP":"tab:red","TDOP":"tab:orange"}
    num = len(next(iter(dops.values())))
    times = times[:num]

    # Compute means for optional dashed lines and summary text
    means = {k: float(np.nanmean(dops[k])) for k in order}

    fig, ax = plt.subplots(figsize=(10,5))
    for k in order:
        ax.plot(times, dops[k], label=k, color=colors[k], lw=1.6)
        if show_means:
            ax.axhline(means[k], color=colors[k], ls='--', lw=1, alpha=0.7)

    ax.set_ylim(bottom=0)
    ax.set_title("DOP Metrics over Time")
    ax.set_xlabel("Time (hh:mm:ss)")
    ax.set_ylabel("DOP")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    fig.autofmt_xdate(rotation=0)

    # Display summary means in a box on the plot
    if show_means:
        txt = "\n".join([f"{k} mean = {means[k]:.2f}" for k in order])
        ax.text(0.98, 0.98, txt, transform=ax.transAxes, ha="right", va="top",
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8), fontsize=9)
    plt.tight_layout()
    plt.show()


# =========================
# 3) Plots (Geographic & ENU)
# =========================
def plot_geographic_overlay(lon_est, lat_est, lon_gt, lat_gt):
    """Plot geographic overlay of estimated vs ground-truth tracks on lon/lat axes.
    Long/lat axes are formatted to DMS strings.
    """
    def lon_fmt(x, _): return deg_to_dms(x, is_lat=False)
    def lat_fmt(y, _): return deg_to_dms(y, is_lat=True)

    plt.figure(figsize=(7,6))
    plt.plot(lon_est, lat_est, 'r-', lw=2, label='Estimated')
    plt.plot(lon_gt,  lat_gt,  'g-', lw=2, label='Ground Truth')

    # Mark the starts/ends of each track
    plt.plot(lon_est[0], lat_est[0], 'ro', ms=8, label='Start (Est)')
    plt.plot(lon_est[-1], lat_est[-1], 'rs', ms=8, label='End (Est)')
    plt.plot(lon_gt[0],  lat_gt[0],  'go', ms=8, label='Start (GT)')
    plt.plot(lon_gt[-1], lat_gt[-1], 'gs', ms=8, label='End (GT)')

    plt.title('GNSS Receiver Trajectory (Geographic View)')
    ax = plt.gca()
    # Use FuncFormatter to show DMS labels
    ax.xaxis.set_major_formatter(FuncFormatter(lon_fmt))
    ax.yaxis.set_major_formatter(FuncFormatter(lat_fmt))
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.grid(True, alpha=0.4); plt.legend(loc='best', framealpha=0.9)
    plt.tight_layout(); plt.show()


def plot_enu_3d(E_est, N_est, U_est, E_gt, N_gt, U_gt):
    """Plot 3D ENU trajectories for estimated and ground-truth positions.
    Uses matplotlib 3D axes.
    """
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection='3d')

    # Trajectories
    ax.plot(E_est, N_est, U_est, 'r-', lw=2, label='Estimated')
    ax.plot(E_gt,  N_gt,  U_gt,  'g-', lw=2, label='Ground Truth')

    # Start points — scatter for emphasis
    ax.scatter(E_est[0], N_est[0], U_est[0], c='r', s=60, label='Start (Est)')
    ax.scatter(E_gt[0],  N_gt[0],  U_gt[0],  c='g', s=60, label='Start (GT)')

    # Labels and layout
    ax.set_title('3D Trajectory (ENU Frame)')
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_zlabel('Up (m)')
    ax.grid(True)
    ax.legend(loc='best', framealpha=0.9)
    plt.tight_layout()
    plt.show()


# =========================
# 4) Error computation + summary
# =========================
def compute_errors(est_ENU, gt_ENU):
    """Compute errors between estimated ENU and ground-truth ENU.
    Returns per-axis errors and combined 2D/3D errors.
    """
    E_est, N_est, U_est = est_ENU
    E_gt,  N_gt,  U_gt  = gt_ENU
    n = min(len(E_est), len(E_gt))  # compare only up to shortest series
    err_E = E_est[:n] - E_gt[:n]
    err_N = N_est[:n] - N_gt[:n]
    err_U = U_est[:n] - U_gt[:n]
    err_2D = np.sqrt(err_E**2 + err_N**2)
    err_3D = np.sqrt(err_E**2 + err_N**2 + err_U**2)
    return err_E, err_N, err_U, err_2D, err_3D


def print_summary(err_2D, err_3D, lat_est, lon_est, alt_est):
    """Print a brief textual summary of the positioning results (2D/3D stats and coordinate ranges)."""
    rms = lambda x: np.sqrt(np.mean(np.asarray(x)**2))
    summary = {
        "2D": dict(Mean=np.mean(err_2D), RMS=rms(err_2D), Std=np.std(err_2D),
                   Max=np.max(err_2D), Min=np.min(err_2D), P95=np.percentile(err_2D,95)),
        "3D": dict(Mean=np.mean(err_3D), RMS=rms(err_3D), Std=np.std(err_3D),
                   Max=np.max(err_3D), Min=np.min(err_3D), P95=np.percentile(err_3D,95)),
    }

    print("\n===========================================")
    print(" GNSS SPP POSITIONING RESULTS SUMMARY")
    print("===========================================")
    print(f"Total Epochs Processed: {len(err_2D)}")
    for k in ["2D","3D"]:
        print(f"\n{k} Error (m):")
        for name, val in summary[k].items():
            print(f"  {name:>12s}: {val:10.3f}")
    print("\nCoordinate ranges:")
    # Print lat/lon/alt ranges for context
    print(f"  Latitude : {lat_est.min():.6f}° to {lat_est.max():.6f}°")
    print(f"  Longitude: {lon_est.min():.6f}° to {lon_est.max():.6f}°")
    print(f"  Altitude : {alt_est.min():.3f} m to {alt_est.max():.3f} m")


# =========================
# 5) Error plots (epoch-based) + Histograms per meter (ignore >1500 m)
# =========================
def plot_error_dashboards(err_E, err_N, err_U, err_2D, err_3D):
    """Create a 2x2 dashboard of per-axis errors and combined 2D/3D plot.
    Uses fill_between to highlight the error envelopes.
    """
    epochs = np.arange(len(err_E))
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    plt.fill_between(epochs, err_E, color='steelblue', alpha=0.35)
    plt.plot(epochs, err_E, color='steelblue')
    plt.axhline(0, color='k', lw=0.8, ls='--')
    plt.title("East Positioning Error"); plt.xlabel("Epoch"); plt.ylabel("East Error (m)")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.fill_between(epochs, err_N, color='tomato', alpha=0.35)
    plt.plot(epochs, err_N, color='tomato')
    plt.axhline(0, color='k', lw=0.8, ls='--')
    plt.title("North Positioning Error"); plt.xlabel("Epoch"); plt.ylabel("North Error (m)")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.fill_between(epochs, err_U, color='mediumseagreen', alpha=0.35)
    plt.plot(epochs, err_U, color='mediumseagreen')
    plt.axhline(0, color='k', lw=0.8, ls='--')
    plt.title("Up Positioning Error"); plt.xlabel("Epoch"); plt.ylabel("Up Error (m)")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    plt.fill_between(epochs, err_2D, color='plum', alpha=0.35, label='2D Error')
    plt.plot(epochs, err_2D, color='purple', label='2D Error')
    plt.fill_between(epochs, err_3D, color='gold', alpha=0.25, label='3D Error')
    plt.plot(epochs, err_3D, color='orange', label='3D Error')
    plt.title("2D and 3D Positioning Errors"); plt.xlabel("Epoch"); plt.ylabel("Error (m)")
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.tight_layout(); plt.show()


def plot_histograms_and_scatter(err_E, err_N, err_2D, err_3D,
                                bin_size=1, max_keep=1500):
    """Show histograms for 2D/3D error distributions (1 m bins by default),
    exclude very large outliers (> max_keep) from histograms but still show
    error evolution and E–N scatter for all values.
    """
    def _clean(x, max_keep):
        x = np.asarray(x); x = x[np.isfinite(x)]
        return x[(x >= 0) & (x <= max_keep)]

    def _upper_limit(x, max_keep):
        if x.size == 0: return 10
        p995 = np.percentile(x, 99.5)
        return float(np.ceil(min(max_keep, max(10, p995))))

    # Prepare histogram ranges
    e2 = _clean(err_2D, max_keep); e3 = _clean(err_3D, max_keep)
    x2_max = _upper_limit(e2, max_keep); x3_max = _upper_limit(e3, max_keep)
    bins_2d = np.arange(0, x2_max + bin_size, bin_size)
    bins_3d = np.arange(0, x3_max + bin_size, bin_size)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    # 2D histogram with mean & RMS lines
    ax[0].hist(e2, bins=bins_2d, color='teal', edgecolor='black', alpha=0.8)
    if e2.size:
        ax[0].axvline(np.mean(e2), color='r', ls='--', lw=1.5, label=f"Mean: {np.mean(e2):.3f} m")
        ax[0].axvline(np.sqrt(np.mean(e2**2)), color='orange', ls='--', lw=1.5, label=f"RMS: {np.sqrt(np.mean(e2**2)):.3f} m")
    ax[0].set_xlim(0, x2_max); ax[0].set_title("2D Error Distribution (1 m bins)")
    ax[0].set_xlabel("2D Error (m)"); ax[0].set_ylabel("Count"); ax[0].legend()

    # 3D histogram with mean & RMS lines
    ax[1].hist(e3, bins=bins_3d, color='slateblue', edgecolor='black', alpha=0.8)
    if e3.size:
        ax[1].axvline(np.mean(e3), color='r', ls='--', lw=1.5, label=f"Mean: {np.mean(e3):.3f} m")
        ax[1].axvline(np.sqrt(np.mean(e3**2)), color='orange', ls='--', lw=1.5, label=f"RMS: {np.sqrt(np.mean(e3**2)):.3f} m")
    ax[1].set_xlim(0, x3_max); ax[1].set_title("3D Error Distribution (1 m bins)")
    ax[1].set_xlabel("3D Error (m)"); ax[1].set_ylabel("Count"); ax[1].legend()
    plt.tight_layout(); plt.show()

    # Evolution (epoch-based) + E–N scatter
    epochs = np.arange(len(err_2D))
    e2_s = moving_average(err_2D, k=9)
    e3_s = moving_average(err_3D, k=9)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    # Raw and smoothed time series
    ax[0].plot(epochs, err_2D, color='plum', alpha=0.35, label='2D Raw')
    ax[0].plot(epochs[:len(e2_s)], e2_s, color='purple', lw=2, label='2D Smoothed')
    ax[0].plot(epochs, err_3D, color='orange', alpha=0.35, label='3D Raw')
    ax[0].plot(epochs[:len(e3_s)], e3_s, color='darkorange', lw=2, label='3D Smoothed')
    ax[0].set_title("Positioning Error Evolution"); ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Positioning Error (m)")
    ax[0].legend(); ax[0].grid(True, alpha=0.3)

    # E–N scatter colored by epoch for temporal context
    sc = ax[1].scatter(err_E, err_N, c=epochs, cmap='viridis', edgecolor='none')
    ax[1].axhline(0, color='gray', lw=0.8, ls='--'); ax[1].axvline(0, color='gray', lw=0.8, ls='--')
    ax[1].set_xlabel("East Error (m)"); ax[1].set_ylabel("North Error (m)")
    ax[1].set_title("East vs North Error Distribution")
    cb = plt.colorbar(sc, ax=ax[1]); cb.set_label("Epoch")
    plt.tight_layout(); plt.show()


# =========================
# 6) Main workflow + CSV exports
# =========================
def main():
    # Load ground truth and estimated solutions from CSVs (filenames expected in working dir)
    gt = load_ground_truth('NAV-HPPOSECEF.csv')
    est = load_estimated_solution('satellite_positions.csv','pseudoranges_meas.csv',
                                  'satellite_clock_bias.csv','ionospheric_delay.csv','tropospheric_delay.csv')

    # --- ENU conversion (use first GT point as local origin)
    ref_xyz = np.array([gt['ecef_x'][0], gt['ecef_y'][0], gt['ecef_z'][0]])
    gt_ecef = np.column_stack([gt['ecef_x'], gt['ecef_y'], gt['ecef_z']])

    # Convert all GT ECEF points into ENU with respect to reference
    E_gt, N_gt, U_gt = [], [], []
    for x, y, z in gt_ecef:
        e, n, u = ecef_to_enu(x, y, z, *ref_xyz)
        E_gt.append(e); N_gt.append(n); U_gt.append(u)
    E_gt, N_gt, U_gt = np.array(E_gt), np.array(N_gt), np.array(U_gt)

    # Convert estimated ECEF positions into the same ENU frame
    E_est, N_est, U_est = [], [], []
    for x, y, z in est['ecef']:
        e, n, u = ecef_to_enu(x, y, z, *ref_xyz)
        E_est.append(e); N_est.append(n); U_est.append(u)
    E_est, N_est, U_est = np.array(E_est), np.array(N_est), np.array(U_est)

    # --- Visualization: geographic overlay and ENU 3D trajectory
    plot_geographic_overlay(est['lon'], est['lat'], gt['lon'], gt['lat'])
    plot_enu_3d(E_est, N_est, U_est, E_gt, N_gt, U_gt)

    # --- DOP vs time (uses satellite positions + estimated ECEF series)
    dops = compute_dops_over_time('satellite_positions.csv', est['ecef'])
    plot_dops_time(dops, 'NAV-HPPOSECEF.csv', utc_offset_hours=0)

    # --- Errors + visuals
    err_E, err_N, err_U, err_2D, err_3D = compute_errors((E_est,N_est,U_est), (E_gt,N_gt,U_gt))
    plot_error_dashboards(err_E, err_N, err_U, err_2D, err_3D)
    plot_histograms_and_scatter(err_E, err_N, err_2D, err_3D, bin_size=1, max_keep=1500)

    # --- Summary printed to console
    print_summary(err_2D, err_3D, est['lat'], est['lon'], est['alt'])

    # --- CSV exports (estimated positions)
    est_out = np.column_stack([
        np.arange(len(est['ecef'])),             # Epoch
        est['lat'], est['lon'], est['alt'],      # LLA
        est['clk_bias_m'],                       # Clock_Bias_m (meters)
        E_est, N_est, U_est,                     # ENU
        est['ecef'][:,0], est['ecef'][:,1], est['ecef'][:,2]  # ECEF
    ])
    np.savetxt("estimated_positions.csv", est_out, delimiter=",",
               header="Epoch,Latitude_deg,Longitude_deg,Altitude_m,Clock_Bias_m,East_m,North_m,Up_m,ECEF_X_m,ECEF_Y_m,ECEF_Z_m",
               comments='', fmt="%.6f")

    # --- CSV exports (errors)
    err_out = np.column_stack([np.arange(len(err_E)), err_E, err_N, err_U, err_2D, err_3D])
    np.savetxt("positioning_errors.csv", err_out, delimiter=",",
               header="Epoch,Err_E_m,Err_N_m,Err_U_m,Err_2D_m,Err_3D_m", comments='', fmt="%.6f")

    print("\nOutput files generated:")
    print("  • estimated_positions.csv")
    print("  • positioning_errors.csv")


if __name__ == "__main__":
    main()
