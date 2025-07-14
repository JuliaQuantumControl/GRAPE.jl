<!--
SPDX-FileCopyrightText: © 2025 Michael Goerz <mail@michaelgoerz.net>

SPDX-License-Identifier: CC-BY-4.0
-->

# JOSS Paper Source

This directory contains the source for the submission to the [Journal of Open Software (JOSS)](https://joss.theoj.org)

See the [JOSS author guide](https://joss.readthedocs.io/en/latest/paper.html) for the format of the paper.

## Compiling Locally

The paper can be compiled locally via `make`. Run `make help` for an overview. Building `paper.pdf` via `make paper` requires [Docker](https://www.docker.com/products/docker-desktop/) or a compatible containerization system to be installed, see below.

See also the [JOSS's "Checking that your paper compiles" instructions](https://joss.readthedocs.io/en/latest/paper.html#checking-that-your-paper-compiles), and the [`openjournals/inara` README](https://github.com/openjournals/inara).


## Quick Preview with `pandoc`

If you cannot set up Docker or a compatible containerization system, or want a quick preview of the manuscript, you can do so directly with [Pandoc](https://pandoc.org) (which JOSS / the Docker image uses under the hood, with heavily customized filters and templates).

Inside the folder containing `paper.md`, run

```
pandoc --citeproc --csl=apa.csl -o paper.pdf --pdf-engine=lualatex paper.md
```

or `make pandoc`.

The resulting file will not have the same formatting as the proper JOSS manuscript, but it will compile much faster to provide a quick preview of the text.


## Automatic Compilation on GitHub Actions

When pushing any change to the files in this folder, the workflow in [`/.github/workflows/JOSS.yml`](https://github.com/JuliaQuantumControl/GRAPE.jl/blob/master/.github/workflows/JOSS.yml) will automatically build the PDF version of the manuscript. The resulting file can be downloaded from the "Artifacts" section of the summary for the run [on GitHub](https://github.com/JuliaQuantumControl/GRAPE.jl/actions/workflows/JOSS.yml).


## Using Podman

The `Makefile` uses `docker` to build the manuscript using a container provided by JOSS. If you do not wish to install Docker on your system (e.g., because you do not have root access), you can also use an alternative containerization system with a command line interface that is compatible with Docker. One attractive alternative is [Podman](https://podman.io).

On macOS with [Homebrew](https://brew.sh):

```
brew install podman
podman machine init
podman machine start
```

You can then compile the manuscript with

```
make DOCKER=podman paper
```

You can also set this as an environment variable (`export DOCKER=podman`) if you want to run `make` repeatedly.

When done, you may run

```
podman machine stop
```

See `podman machine` for more details on how to manage Podman machines.
