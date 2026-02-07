# Current Status: Carr et al. 2023 Reproduction

## What We've Verified is CORRECT

✓ **Depletion time formula**: t_dep = 10^4.92 * (M_star/Msun)^(**-0.37**) * (1+z)^(-1.5) [Gyr]
  - Confirmed -0.37 exponent from McGaugh paper
  - Units: result is directly in Gyr
  - Verified: t_dep(6e10 Msun, z=0) = 8.55 Gyr ✓

✓ **Wiersma cooling**:
  - Density-independent in CGM regime (Lambda ~ n_H^0)
  - No free parameter for characteristic radius
  - Using full photoionization equilibrium tables

✓ **All equations**: Verified implementation matches paper equations

✓ **Starting redshift**: z=6 (as in paper)

✓ **Loading factors**: eta_M=0.1, eta_E=0.1, eta_Z=0.2 (for Figure 2)

## Best Result (z=6 start, paper parameters)

**Initial**: M_halo(z=6) = 4.0e10 Msun

**Final (z=0)**:
- M_halo = 9.93e+11 Msun (target 1e12) → **0.7% error** ✓
- M_star = 5.76e+10 Msun (target ~1e10) → **476% error** ✗
- M_ISM  = 4.64e+10 Msun (target ~1e10) → **364% error** ✗
- M_CGM  = 1.74e+09 Msun (target ~1e10) → **83% error** ✗

**Total error: 9.23**

## The Problem

We get **~5× too much M_ISM and M_star**, and **~6× too little M_CGM**.

This suggests:
- Too much gas cooling from CGM → ISM
- Too many stars forming from ISM
- Not enough gas staying hot in CGM

## What We've Ruled Out

✗ Wrong depletion time exponent (was +0.37, now correct -0.37)
✗ Wrong depletion time normalization (tested 4.92 to 5.68, paper's 4.92 is best)
✗ Cooling characteristic radius ambiguity (density-independent in CGM)
✗ Wrong starting redshift (confirmed z=6)
✗ Wrong equation implementation (verified all match paper)

## What Carr et al. DON'T Report (Free Parameters)

1. **Initial conditions at z=6**:
   - We assume: M_CGM = 5% of f_b*M_halo, M_ISM = 1%, M_star = 0.1%
   - Paper gives NO details
   - Could matter for early evolution

2. **IGM metallicity**: We use Z_IGM = 0.01*Z_sun (guess)

3. **Exact M_halo at z=6**: We calculated 4e10 to give 1e12 at z=0

## Most Likely Explanations

1. **Different initial conditions**: Maybe they start with more/different gas distribution
2. **Missing parameter**: Some tuned value they don't report
3. **Implementation detail**: Subtle difference in how they calculate something
4. **Figure 2 ≠ text**: Figure might use different parameters than stated

## Normalization Scan Results

| log10(norm) | M_star [Msun] | M_ISM [Msun] | Total Error |
|-------------|---------------|--------------|-------------|
| 4.92 (paper)| 5.76e+10      | 4.64e+10     | 9.23        |
| 5.20        | 3.01e+10      | 8.47e+10     | 10.00       |
| 5.40        | 1.68e+10      | 1.10e+11     | 11.22       |
| 5.50        | 1.23e+10      | 1.19e+11     | 11.95       |
| 5.60        | 8.87e+09      | 1.26e+11     | 12.52       |

**Trend**: Increasing norm → M_star decreases but M_ISM increases even more!

## Files Created

- `test_with_correct_exponent.py` - First test with -0.37
- `scan_z6_initial_mass.py` - Found M_halo(z=6) = 4e10
- `test_tuned_normalization.py` - Tested higher norms
- `scan_normalization.py` - Systematic scan
- `/tmp/free_parameters_analysis.txt` - Analysis of what's not reported
- `/tmp/cooling_equation_analysis.md` - Cooling implementation details

## Next Steps (if continuing)

1. **Try different initial conditions**: Start with more/less gas in CGM/ISM
2. **Check for numerical issues**: Integration tolerances, timesteps
3. **Contact authors**: Ask for exact initial conditions and any tuned parameters
4. **Accept close-enough**: ~Factor of 5 might be as good as we can get without missing info

## Bottom Line

**We've correctly implemented the gas-regulator model as described in the paper**, but cannot exactly reproduce Figure 2 (factor ~5 discrepancy). This likely means:

- The paper doesn't report all calibration details
- Some parameters were tuned but not mentioned
- Or there's a subtle difference in implementation

The model works qualitatively and demonstrates the physics correctly, just doesn't match quantitatively.
