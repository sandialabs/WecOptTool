## A Visual Exploration of Optimal Control in heaving WECs

### Daniel Gaebele, Giorgio Bacelli, Ryan Coe, Jeff Grasberger, Carlos Michelen Strofer
#### Sandia National Laboratories

initially presented at UMERC 2025, Corvallis, OR (and since improved)

This branch is intended to help users re-create visuals and animations like Sankey average power flows diagrams, WEC power curves, and dynamic animations including the phase space.
Some example figures and animations are contained in the [`gfx`](/gfx/) folder.


It is strongly recommended that you create a dedicated virtual environment based on the `environment.yaml` file using [`conda`](https://www.anaconda.com/).

```bash
conda env create --file environment.yaml
```

Use your new environemnt `wot_3p1_umerc2025` to locally run the notebook `Umerc_2025_wavebot_power_curves_and_animations.ipynb`, which will show you how to use the custom functions contained in the source file `umerc2025_utils.py` to generate Sankey average power flows diagrams, WEC power curves, and custom animations using WecOptTool results.

