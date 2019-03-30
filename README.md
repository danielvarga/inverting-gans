# Inverting GANs left and right

Hardwired constants you might want to change:

- in `dcgan.py`: `self.latent_dim` and `epochs`,
- in `inverter.py`: `latent_dim`, `epochs` and `which` (either `"hourglass"` for right inverse or `"barrel"` for left inverse).
