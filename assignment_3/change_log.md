# LBM for Karman vortex street change log

## Initial implementation of LBM for Karman vortex street simulation based on course demo:
This works, but at Re = 150, noise forms at the inlet. This was fixed by using 1.5x the grid points in both directions.
I will try to increase grid points and reduce velocity to get a higher Re number. The main factor is tau, the relaxation time.

## Increased grid points:
I changed the drid to 440 x 82 (0.05cm resolution), which gives Re = 200.

## Increased grid points and optimized code:
I changed the grid to 880 x 164 (0.025cm resolution), which gives Re = 400. To get results in a reasonable time, I optimized the code using numbas njit.


## Initialization improvements:
Ramp up the velocity at the inlet using an exponential function to reduce shocks. We also now use a smooth initalization of the velocity field to reduce initial transients.
I managed to get Re = 450 for a bit, and then the simulation broke (at the inlet).

## New inlet BC:
I implemented a regularized Zou-He inlet BC, which reduces shocks and allows me to reach Re = 500 without breaking. 
I even managed to go up to Re = 700 and inlet instabilities are not an issues. The issue is that instabilities develop at the top and bottom boundaries, in the high velocity region between two vortexes.

## Fixed wrong top and bottom BC:
It turns out that I was using periodic BCs at the top and bottom, which are more stable. After changing to bounce back we can get Re = 500.

## Added non-equillibrium damping:
I added damping in the collision step to reduce non-equilibrium effects that can cause instabilities. This allows me to reach Re = 600 without breaking.
Although, it is very close to breaking.

## Magical changes:
After some magical changes, it now reaches Re = 850. No idea why or how. I tried to add trt functionality, which didn't work well, but after trying bgk again it works better


## Further improvements to reduce noise and increase stability:
| Category     | Method                              | Effect                                      |
| ------------ | ----------------------------------- | ------------------------------------------- |
| Inlet BC     | Regularized Zou-He                  | Reduce inlet shocks                         |
| Outlet BC    | Convective / sponge                 | Reduce reflections of vortices              |
| Collision    | MRT / TRT / Entropic                | Increase stability at low ν                 |
| Obstacle BC  | Half-way / interpolated bounce-back | Reduce velocity spikes near cylinder        |
| Filtering    | Regularized LBM, low-pass filter    | Remove spurious high-frequency oscillations |
| Mach control | Reduce effective Mach / timestep    | Reduce compressibility noise                |
