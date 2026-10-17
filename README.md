# CFBoxes

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://dubosipsl.github.io/CFBoxes.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://dubosipsl.github.io/CFBoxes.jl/dev/)
[![Build Status](https://github.com/dubosipsl/CFBoxes.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/dubosipsl/CFBoxes.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/dubosipsl/CFBoxes.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/dubosipsl/CFBoxes.jl)

## Layout

Each discretization is a submodule of `CFBoxes`, in its own directories, and `CFBoxes` re-exports its exported names,
so `using CFBoxes` gives all of them:

| Submodule | Source | Tests | Contents |
|---|---|---|---|
| `CFBoxes.CGrid` | `src/CGrid/` | `test/CGrid/` | 2D and 3D boxes on a C-grid: grids with halos, operators, wall boundary conditions, vortex-force schemes |

To add a submodule `Name`:
1. put its code in `src/Name/`, with `src/Name/Name.jl` defining `module Name` and its exports;
2. in `src/CFBoxes.jl`, include it, add `using .Name`, and re-export its names as for `CGrid`;
3. put its tests in `test/Name/`, include `test/Name/runtests.jl` from `test/runtests.jl`, and add `Name` to the
   submodules of the re-export check there;
4. add a section for it in `docs/src/index.md` and add it to `modules` in `docs/make.jl`.

Two submodules must not export the same name: in `CFBoxes` it would be ambiguous, and fail when used.
