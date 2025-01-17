# Setup

# Env
- DON'T use conda as it prefers it's on ld path and libstd etc
- Remaining instructions are for Ubuntu only

```bash
sudo apt install python3-pip python3-venv python3-dev
python3 -m venv ${HOME}/.venv/nft
```

# Arrow C++ libs

As a shortcut we won't build the arrow c++ libs from scratch.
Instead, follow the [arrow installation instructions](https://arrow.apache.org/install/).
These instructions don't include `libarrow-python-dev` so that needs to be apt installed
separately.

# Build pyarrow

Assume CWD is where you want to put the repo:

```bash
source ${HOME}/.venv/nft/bin/activate
cd /path/to/lance/python/thirdparty
./build.sh
```

Make sure pyarrow works properly:

```python
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
```

# Build Lance

Assume CWD is where you want to put the repo:

```bash
git clone git@github.com:eto-ai/lance

pushd nft/cpp
cmake . -B build
pushd build
make -j
popd
popd

pushd python
source ${HOME}/.venv/nft/bin/activate
python setup.py develop
```

Test the installation in python:

```python
import duckdb
import pylance
uri = "..../pet.lance"
pets = pylance.dataset(uri)
duckdb.query('select label, count(1) from pets group by label').to_arrow_table()
```