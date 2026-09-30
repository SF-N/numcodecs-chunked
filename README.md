[![image](https://img.shields.io/github/actions/workflow/status/SF-N/numcodecs-chunked/ci.yml?branch=main)](https://github.com/SF-N/numcodecs-chunked/actions/workflows/ci.yml?query=branch%3Amain)
[![image](https://img.shields.io/pypi/v/numcodecs-chunked.svg)](https://pypi.python.org/pypi/numcodecs-chunked)
[![image](https://img.shields.io/pypi/l/numcodecs-chunked.svg)](https://github.com/SF-N/numcodecs-chunked/blob/main/LICENSE)
[![image](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FSF-N%2Fnumcodecs-chunked%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)](https://pypi.python.org/pypi/numcodecs-chunked)
[![image](https://readthedocs.org/projects/numcodecs-chunked/badge/?version=latest)](https://numcodecs-chunked.readthedocs.io/en/latest/?badge=latest)

# numcodecs-chunked

`ChunkedCodec` for the [`numcodecs`] buffer compression API.

The `ChunkedCodec` is a meta-codec that splits an array into chunks and applies an inner codec to each chunk independently. The `chunk_shape` has one entry per axis: a positive integer chunk length, `None` for the entire axis, or the sentinel `"..."` (used at most once) for all remaining axes.

```python
from numcodecs_chunked import ChunkedCodec

# compress every 2D field of a (time, lat, lon) array independently
codec = ChunkedCodec(codec=dict(id="zlib", level=9), chunk_shape=[1, "..."])

# compress 100x100 tiles that span the entire first axis
codec = ChunkedCodec(codec=dict(id="zlib", level=9), chunk_shape=[None, 100, 100])
```

[`numcodecs`]: https://numcodecs.readthedocs.io/en/stable/

## License

Licensed under the Mozilla Public License, Version 2.0 ([LICENSE](LICENSE) or https://www.mozilla.org/en-US/MPL/2.0/).


## Funding

The `numcodecs-chunked` package has been developed as part of [ESiWACE3](https://www.esiwace.eu), the third phase of the Centre of Excellence in Simulation of Weather and Climate in Europe.

Funded by the European Union. This work has received funding from the European High Performance Computing Joint Undertaking (JU) under grant agreement No 101093054.
