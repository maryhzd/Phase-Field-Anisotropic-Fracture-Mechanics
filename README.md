# Constants Summary

This document summarizes the constants used in the phase-field calculations for anisotropic fracture mechanics as detailed in the related paper.

## General Constants

| Constant      | Value               | Description                             |
|---------------|---------------------|-----------------------------------------|
| `eta_epsilon` | `0.001`             | Small stability parameter               |
| `beta_1`      | `0.8`               | Coeff 1 for equation of crack normal    |
| `beta_2`      | `0.001`             | Coeff 2 for equation of crack normal    |
| `alpha`       | `-0.99`             | Anisotropy parameter 1                  |
| `alpha_1`     | `-0.99`             | Anisotropy parameter 2                  |
| `epsilon`     | `0.04`              | Regularization parameter                |
| `g_m`         | `10`                | Critical Gradient of Phi magnitude      |
| `l_1`         | `0.01`              | Length scale parameter 1                |
| `phi_m`       | `0.98`              | Critical phase-field value              |
| `l_2`         | `0.001`             | Length scale parameter 2                |
| `m`           | `4e5 GPa`           | Large penalty parameter                 |

## Material Properties used for Homogenization:

### Material A (Strong Material)

| Property      | Value               | Description                             |
|---------------|---------------------|-----------------------------------------|
| `E`           | `21.7 GPa`          | Young's modulus                         |
| `nu`          | `0.23`              | Poisson's ratio                         |

### Material B (Weak Material)

| Property      | Value               | Description                             |
|---------------|---------------------|-----------------------------------------|
| `E`           | `11.9 GPa`          | Young's modulus                         |
| `nu`          | `0.21`              | Poisson's ratio                         |

### Homogenized 'G_c' value:

| Parameter     | Value               | Description                                      |
|---------------|---------------------|--------------------------------------------------|
| `G_c`         | `4e4 N/m`           | Fracture toughness                               |



