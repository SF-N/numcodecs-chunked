import numcodecs
import numcodecs.registry
import numpy as np
import pytest

INNER = dict(id="zlib", level=1)


def test_from_config():
    codec = numcodecs.registry.get_codec(
        dict(id="chunked", codec=INNER, chunk_shape=[1, None, "..."])
    )
    assert codec.__class__.__name__ == "ChunkedCodec"
    assert codec.__class__.__module__ == "numcodecs_chunked"
    assert codec.get_config() == dict(
        id="chunked", codec=INNER, chunk_shape=[1, None, "..."]
    )


def test_ellipsis():
    from numcodecs_chunked import ChunkedCodec

    codec = ChunkedCodec(codec=INNER, chunk_shape=[1, ...])
    assert codec.get_config()["chunk_shape"] == [1, "..."]


def test_invalid():
    from numcodecs_chunked import ChunkedCodec

    with pytest.raises(ValueError):
        ChunkedCodec(codec=INNER, chunk_shape=[0])
    with pytest.raises(ValueError):
        ChunkedCodec(codec=INNER, chunk_shape=["...", 1, "..."])
    with pytest.raises(TypeError):
        ChunkedCodec(codec=INNER, chunk_shape=[1.5])
    with pytest.raises(ValueError):
        ChunkedCodec(codec=INNER, chunk_shape=[1, 1]).encode(np.zeros((2, 2, 2)))
    with pytest.raises(ValueError):
        ChunkedCodec(codec=INNER, chunk_shape=[1, 1, "..."]).encode(np.zeros((2,)))


def test_map():
    from numcodecs_chunked import ChunkedCodec

    codec = ChunkedCodec(codec=INNER, chunk_shape=[1, "..."])
    mapped = codec.map(lambda c: numcodecs.registry.get_codec(dict(id="zlib", level=9)))
    assert mapped.get_config()["codec"] == dict(id="zlib", level=9)


def check_roundtrip(data: np.ndarray, chunk_shape):
    codec = numcodecs.registry.get_codec(
        dict(id="chunked", codec=INNER, chunk_shape=chunk_shape)
    )

    encoded = codec.encode(data)
    decoded = np.asarray(codec.decode(encoded))

    assert decoded.dtype == data.dtype
    assert decoded.shape == data.shape
    np.testing.assert_array_equal(decoded, data)

    out = np.empty_like(data)
    codec.decode(encoded, out=out)
    np.testing.assert_array_equal(out, data)


def test_roundtrip():
    data = np.arange(3 * 7 * 11, dtype=np.float64).reshape(3, 7, 11)
    check_roundtrip(data, [1, "..."])
    check_roundtrip(data, [1, None, None])
    check_roundtrip(data, [None, 3, 4])
    check_roundtrip(data, [2, 2, 2])
    check_roundtrip(data, ["..."])
    check_roundtrip(data, [None, "...", 5])
    check_roundtrip(data.astype(np.int16), [1, 7, 11])
    check_roundtrip(np.zeros((0, 5)), [1, "..."])
    check_roundtrip(np.zeros((), dtype=np.float32), ["..."])
