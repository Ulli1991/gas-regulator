"""
Wiersma et al. (2009) cooling function interpolator.

Loads the photoionization equilibrium cooling tables and provides
Lambda(T, Z, z) interpolation for the gas-regulator model.
"""

import numpy as np
import h5py
from scipy.interpolate import RegularGridInterpolator
import glob
import os

class WiersmaCooling:
    """
    Interpolator for Wiersma et al. (2009) cooling tables.

    Uses photoionization equilibrium tables with Haardt & Madau UV background.
    """

    def __init__(self, table_dir='data/cooling_tables/CoolingTables'):
        """
        Initialize cooling table interpolator.

        Args:
            table_dir: Path to directory containing z_*.hdf5 files
        """
        self.table_dir = table_dir
        self.Z_sun = 0.0134  # Solar metallicity (Asplund 2009)
        self.He_mass_frac = 0.25  # Assumed helium mass fraction

        print(f"Loading Wiersma cooling tables from {table_dir}...")
        self._load_tables()
        print(f"  Loaded {len(self.redshifts)} redshift tables")
        print(f"  Temperature range: {self.T_bins.min():.1e} - {self.T_bins.max():.1e} K")
        print(f"  Density range: {self.nH_bins.min():.1e} - {self.nH_bins.max():.1e} cm^-3")
        print(f"  Redshift range: {self.redshifts.min():.2f} - {self.redshifts.max():.2f}")

    def _load_tables(self):
        """Load all cooling tables into memory."""
        # Find all redshift files
        files = sorted(glob.glob(os.path.join(self.table_dir, 'z_*.hdf5')))
        files = [f for f in files if 'nocompton' not in f and
                                      'collis' not in f and
                                      'photodis' not in f]

        if len(files) == 0:
            raise FileNotFoundError(f"No cooling tables found in {self.table_dir}")

        # Load first file to get grid dimensions
        with h5py.File(files[0], 'r') as f:
            self.T_bins = f['/Total_Metals/Temperature_bins'][:]
            self.nH_bins = f['/Total_Metals/Hydrogen_density_bins'][:]
            n_T = len(self.T_bins)
            n_nH = len(self.nH_bins)
            n_z = len(files)

        # Allocate arrays for all redshifts
        self.redshifts = np.zeros(n_z)
        self.Lambda_metals = np.zeros((n_z, n_T, n_nH))  # (z, T, nH)
        self.Lambda_H_He = np.zeros((n_z, n_T, n_nH))    # (z, T, nH)

        # Load all tables
        for i, fname in enumerate(files):
            with h5py.File(fname, 'r') as f:
                self.redshifts[i] = f['/Header/Redshift'][0]

                # Metal cooling (Total_Metals group)
                # Shape: (n_T, n_nH)
                self.Lambda_metals[i] = f['/Total_Metals/Net_cooling'][:]

                # H+He cooling (Metal_free group)
                # Shape: (n_He_frac, n_T, n_nH)
                Lambda_mf_full = f['/Metal_free/Net_Cooling'][:]
                He_frac_bins = f['/Metal_free/Helium_mass_fraction_bins'][:]

                # Use i_X_He = -3 (index 4, He=0.4) to match Fielding & Bryan (2022)
                self.Lambda_H_He[i] = Lambda_mf_full[-3, :, :]

        # Convert to log space for interpolation (more accurate)
        self.log_T_bins = np.log10(self.T_bins)
        self.log_nH_bins = np.log10(self.nH_bins)
        self.log_z_bins = np.log10(self.redshifts + 1e-3)  # Avoid log(0)

        # Create interpolators
        # Note: We interpolate Lambda itself, not log(Lambda), because
        # Lambda can be negative (heating from UV background)
        self._interp_metals = RegularGridInterpolator(
            (self.log_z_bins, self.log_T_bins, self.log_nH_bins),
            self.Lambda_metals,
            bounds_error=False,
            fill_value=None  # Extrapolate
        )

        self._interp_H_He = RegularGridInterpolator(
            (self.log_z_bins, self.log_T_bins, self.log_nH_bins),
            self.Lambda_H_He,
            bounds_error=False,
            fill_value=None
        )

    def cooling_rate(self, T, nH, Z, z):
        """
        Get cooling rate Lambda [erg cm^3 s^-1].

        Implements WSS08 equation 4:
        Lambda_net = Lambda_H,He + (Z/Z_sun) * Lambda_metals

        Args:
            T: Temperature [K] (scalar or array)
            nH: Hydrogen number density [cm^-3] (scalar or array)
            Z: Metallicity [absolute mass fraction] (scalar or array)
            z: Redshift (scalar or array)

        Returns:
            Lambda: Cooling rate [erg cm^3 s^-1]
                   Note: Can be negative (heating from UV background)
        """
        # Convert to log space
        log_T = np.log10(np.maximum(T, self.T_bins.min()))
        log_nH = np.log10(np.maximum(nH, self.nH_bins.min()))
        log_z = np.log10(np.maximum(z, 0.001) + 1e-3)

        # Clip to table bounds
        log_T = np.clip(log_T, self.log_T_bins.min(), self.log_T_bins.max())
        log_nH = np.clip(log_nH, self.log_nH_bins.min(), self.log_nH_bins.max())
        log_z = np.clip(log_z, self.log_z_bins.min(), self.log_z_bins.max())

        # Interpolate (returns Lambda/n_H^2 [erg s^-1 cm^3])
        pts = np.column_stack([log_z, log_T, log_nH]) if np.ndim(T) > 0 else \
              np.array([[log_z, log_T, log_nH]])

        Lambda_H_He_norm = self._interp_H_He(pts)
        Lambda_metals_norm = self._interp_metals(pts)

        # Convert scalar outputs back to scalars
        if np.ndim(T) == 0:
            Lambda_H_He_norm = float(Lambda_H_He_norm)
            Lambda_metals_norm = float(Lambda_metals_norm)

        # Combine contributions (WSS08 eq. 4)
        # The tables give Lambda/n_H^2, which is already the cooling function
        Lambda_net = Lambda_H_He_norm + (Z / self.Z_sun) * Lambda_metals_norm

        # Ensure non-negative (can be slightly negative due to interpolation)
        # Actually, Lambda CAN be negative (heating from UV background), so don't floor
        return Lambda_net

    def __call__(self, T, Z, z=0):
        """
        Simplified interface that computes typical CGM density.

        For use in physics.py cooling_function() interface.

        Args:
            T: Temperature [K]
            Z: Metallicity [absolute mass fraction]
            z: Redshift (default 0)

        Returns:
            Lambda: Cooling rate [erg cm^3 s^-1]
        """
        # Estimate typical CGM hydrogen density
        # For CGM at ~0.1 R_vir, n_H ~ 1e-4 cm^-3 is typical
        # This is a rough approximation; ideally we'd pass density explicitly
        nH_typical = 1e-4  # cm^-3

        return self.cooling_rate(T, nH_typical, Z, z)


# Singleton instance (loaded once)
_wiersma_cooling = None

def get_wiersma_cooling(table_dir='data/cooling_tables/CoolingTables'):
    """Get or create Wiersma cooling interpolator."""
    global _wiersma_cooling
    if _wiersma_cooling is None:
        _wiersma_cooling = WiersmaCooling(table_dir)
    return _wiersma_cooling
