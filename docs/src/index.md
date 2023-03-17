```@meta
CurrentModule = GRAPE
```

# GRAPE.jl

```@eval
using Markdown
using Pkg

VERSION = Pkg.dependencies()[Base.UUID("6b52fcaf-80fe-489a-93e9-9f92080510be")].version

github_badge = "[![Github](https://img.shields.io/badge/JuliaQuantumControl-GRAPE.jl-blue.svg?logo=github)](https://github.com/JuliaQuantumControl/GRAPE.jl)"

version_badge = "![v$VERSION](https://img.shields.io/badge/version-v$VERSION-green.svg)"

Markdown.parse("$github_badge $version_badge")
```

Implementation of ([second-order](https://arxiv.org/abs/1102.4096)) GRadient Ascent Pulse Engineering (GRAPE) [KhanejaJMR2005](@cite)[FouquieresJMR2011](@cite) extended with semi-automatic differentiation [GoerzQ2022](@cite).

Part of [`QuantumControl.jl`](https://github.com/JuliaQuantumControl/QuantumControl.jl#readme) and the [JuliaQuantumControl](https://github.com/JuliaQuantumControl) organization.

## Contents

### Overview

```@contents
Pages = [
    "overview.md",
]
Depth = 1
```

### Examples

```@contents
Pages = [
    "examples/simple_state_to_state.md",
    "examples/perfect_entanglers.md",
]
Depth = 1
```

See also the [general examples](https://juliaquantumcontrol.github.io/QuantumControl.jl/stable/examples/) of the [QuantumControl](https://juliaquantumcontrol.github.io/QuantumControl.jl/stable/) package.

### API

```@contents
Pages = [
    "api.md",
]
Depth = 1
```

## History

See the [Releases](https://github.com/JuliaQuantumControl/GRAPE.jl/releases) on Github.
