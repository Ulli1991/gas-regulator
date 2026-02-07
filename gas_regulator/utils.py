"""
Utility functions for plotting and analysis.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_evolution(result, save_path=None):
    """
    Plot the evolution of a single halo (reproduce Figure 2 style).

    Args:
        result: Result dictionary from run_single_halo
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Halo Evolution: M_halo(z=0) = {result['M_halo'][-1]:.2e} Msun",
                 fontsize=14)

    z = result["redshift"]
    t = result["time"]

    # Plot vs redshift (top row) and time (bottom row)

    # Mass evolution
    ax = axes[0, 0]
    ax.semilogy(z, result["M_halo"], label="M_halo", lw=2)
    ax.semilogy(z, result["M_CGM"], label="M_CGM", lw=2)
    ax.semilogy(z, result["M_ISM"], label="M_ISM", lw=2)
    ax.semilogy(z, result["M_star"], label="M_star", lw=2)
    ax.set_xlabel("Redshift")
    ax.set_ylabel("Mass [Msun]")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    ax.set_title("Mass Evolution")

    ax = axes[1, 0]
    ax.semilogy(t, result["M_halo"], label="M_halo", lw=2)
    ax.semilogy(t, result["M_CGM"], label="M_CGM", lw=2)
    ax.semilogy(t, result["M_ISM"], label="M_ISM", lw=2)
    ax.semilogy(t, result["M_star"], label="M_star", lw=2)
    ax.set_xlabel("Time [Gyr]")
    ax.set_ylabel("Mass [Msun]")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # CGM temperature
    ax = axes[0, 1]
    ax.semilogy(z, result["T_CGM"], label="T_CGM", lw=2, color='red')
    ax.set_xlabel("Redshift")
    ax.set_ylabel("Temperature [K]")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    ax.set_title("CGM Temperature")

    ax = axes[1, 1]
    ax.semilogy(t, result["T_CGM"], label="T_CGM", lw=2, color='red')
    ax.set_xlabel("Time [Gyr]")
    ax.set_ylabel("Temperature [K]")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # CGM metallicity
    ax = axes[0, 2]
    ax.plot(z, result["Z_CGM"], label="Z_CGM/Z_sun", lw=2, color='green')
    ax.set_xlabel("Redshift")
    ax.set_ylabel("Z/Z_sun")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    ax.set_title("CGM Metallicity")

    ax = axes[1, 2]
    ax.plot(t, result["Z_CGM"], label="Z_CGM/Z_sun", lw=2, color='green')
    ax.set_xlabel("Time [Gyr]")
    ax.set_ylabel("Z/Z_sun")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return fig


def plot_scaling_relations(results, save_path=None):
    """
    Plot scaling relations from halo suite (reproduce Figures 3-5 style).

    Args:
        results: List of result dictionaries from run_halo_suite
        save_path: Optional path to save figure
    """
    # Extract final values (z=0)
    M_halo_arr = []
    M_star_arr = []
    M_ISM_arr = []
    M_CGM_arr = []
    Z_CGM_arr = []

    for result in results:
        if result is not None:
            M_halo_arr.append(result["M_halo"][-1])
            M_star_arr.append(result["M_star"][-1])
            M_ISM_arr.append(result["M_ISM"][-1])
            M_CGM_arr.append(result["M_CGM"][-1])
            Z_CGM_arr.append(result["Z_CGM"][-1])

    M_halo_arr = np.array(M_halo_arr)
    M_star_arr = np.array(M_star_arr)
    M_ISM_arr = np.array(M_ISM_arr)
    M_CGM_arr = np.array(M_CGM_arr)
    Z_CGM_arr = np.array(Z_CGM_arr)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Scaling Relations at z=0", fontsize=14)

    # M_star / M_halo vs M_halo (Figure 3 style)
    ax = axes[0, 0]
    ax.loglog(M_halo_arr, M_star_arr / M_halo_arr, 'o-', lw=2, ms=6)
    ax.set_xlabel("M_halo [Msun]")
    ax.set_ylabel("M_star / M_halo")
    ax.grid(True, alpha=0.3)
    ax.set_title("Stellar-to-Halo Mass Fraction")

    # M_ISM / M_star vs M_star (Figure 4 style)
    ax = axes[0, 1]
    ax.loglog(M_star_arr, M_ISM_arr / M_star_arr, 'o-', lw=2, ms=6, color='orange')
    ax.set_xlabel("M_star [Msun]")
    ax.set_ylabel("M_ISM / M_star")
    ax.grid(True, alpha=0.3)
    ax.set_title("Gas-to-Stellar Mass Fraction")

    # M_CGM / M_halo vs M_halo (Figure 5 style)
    ax = axes[1, 0]
    ax.loglog(M_halo_arr, M_CGM_arr / M_halo_arr, 'o-', lw=2, ms=6, color='purple')
    ax.set_xlabel("M_halo [Msun]")
    ax.set_ylabel("M_CGM / M_halo")
    ax.grid(True, alpha=0.3)
    ax.set_title("CGM-to-Halo Mass Fraction")

    # Z_CGM vs M_halo (Figure 5 style)
    ax = axes[1, 1]
    ax.semilogx(M_halo_arr, Z_CGM_arr, 'o-', lw=2, ms=6, color='green')
    ax.set_xlabel("M_halo [Msun]")
    ax.set_ylabel("Z_CGM / Z_sun")
    ax.grid(True, alpha=0.3)
    ax.set_title("CGM Metallicity")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return fig


def print_summary(result):
    """
    Print summary statistics for a single halo result.

    Args:
        result: Result dictionary from run_single_halo
    """
    print("\n" + "="*60)
    print("HALO EVOLUTION SUMMARY")
    print("="*60)

    print(f"\nInitial conditions (z={result['redshift'][0]:.2f}):")
    print(f"  M_halo = {result['M_halo'][0]:.2e} Msun")
    print(f"  M_CGM  = {result['M_CGM'][0]:.2e} Msun")
    print(f"  M_ISM  = {result['M_ISM'][0]:.2e} Msun")
    print(f"  M_star = {result['M_star'][0]:.2e} Msun")
    print(f"  T_CGM  = {result['T_CGM'][0]:.2e} K")
    print(f"  Z_CGM  = {result['Z_CGM'][0]:.3f} Z_sun")

    print(f"\nFinal conditions (z={result['redshift'][-1]:.2f}):")
    print(f"  M_halo = {result['M_halo'][-1]:.2e} Msun")
    print(f"  M_CGM  = {result['M_CGM'][-1]:.2e} Msun")
    print(f"  M_ISM  = {result['M_ISM'][-1]:.2e} Msun")
    print(f"  M_star = {result['M_star'][-1]:.2e} Msun")
    print(f"  T_CGM  = {result['T_CGM'][-1]:.2e} K")
    print(f"  Z_CGM  = {result['Z_CGM'][-1]:.3f} Z_sun")

    print(f"\nFinal fractions:")
    print(f"  M_star / M_halo = {result['M_star'][-1] / result['M_halo'][-1]:.4f}")
    print(f"  M_ISM / M_star  = {result['M_ISM'][-1] / result['M_star'][-1]:.4f}")
    print(f"  M_CGM / M_halo  = {result['M_CGM'][-1] / result['M_halo'][-1]:.4f}")
    print(f"  f_baryon total  = {(result['M_star'][-1] + result['M_ISM'][-1] + result['M_CGM'][-1]) / result['M_halo'][-1]:.4f}")

    print("\n" + "="*60)
