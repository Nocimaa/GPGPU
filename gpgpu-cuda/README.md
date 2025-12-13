0. If you're using Nix on the OpenStack, use the provided flake.

```
nix develop
```

1. Build the project (in Debug or Release) with cmake

```
export buildir=... # pas dans l'AFS
cmake -S . -B $builddir -DCMAKE_BUILD_TYPE=Debug
```

or

```
cmake -S . -B $buildir -DCMAKE_BUILD_TYPE=Release
```

2.
Run with

```
$buildir/stream --mode=[gpu,cpu] <video.mp4> [--output=output.mp4]
```

You can also tweak the runtime options to align with the parameters described in `AGENTS - CUDA.md`:
```
--bg=<path>               # use a provided background image instead of estimating it
--opening_size=<int>      # kernel size for the opening/closing steps (default 3)
--th_low=<int>            # low threshold for hysteresis (default 3)
--th_high=<int>           # high threshold for hysteresis (default 30)
--bg_sampling_rate=<ms>   # milliseconds between background samples (default 500)
--bg_number_frame=<frames># number of samples in the background average (default 10)
```

3.
Edit your cuda/cpp code in */Compute.*
