import numpy as np

# Prior on lift (normal): mean 0.10, sd 0.05  → variance 0.0025
mu0, var0 = 0.10, 0.05**2

# New estimate from a quick geo test: 0.12 with sd 0.04 → variance 0.0016
dbar, vard = 0.12, 0.04**2

# Precision-weighted posterior
prec0, precd = 1/var0, 1/vard
mu_post = (prec0*mu0 + precd*dbar) / (prec0 + precd)
var_post = 1 / (prec0 + precd)

print("Posterior mean lift: %.4f (%.2f%%)" % (mu_post, 100*mu_post))
print("Posterior sd: %.4f" % np.sqrt(var_post))
