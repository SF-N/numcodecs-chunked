"""
[`ChunkedCodec`][numcodecs_chunked.ChunkedCodec] for the [`numcodecs`][numcodecs] buffer compression API.
"""

__all__ = ["ChunkedCodec"]

import itertools
from collections.abc import Callable, Sequence
from functools import reduce
from io import BytesIO
from types import EllipsisType

import leb128
import numcodecs.compat
import numcodecs.registry
import numpy as np
from numcodecs.abc import Codec
from numcodecs_combinators.abc import CodecCombinatorMixin
from typing_extensions import Buffer  # MSPV 3.12

ALL_OTHER_AXES = "..."
""" Sentinel that stands for all remaining axes in a chunk shape. """


class ChunkedCodec(Codec, CodecCombinatorMixin):
    """
    Meta-codec that splits an array into chunks and applies the `codec` to
    each chunk independently.

    The `chunk_shape` has one entry per axis of the array. Each entry is either

    - a positive integer, the chunk length along this axis (the last chunk may
      be shorter if the axis length is not a multiple of the chunk length),
    - [`None`][None], meaning that the chunk spans the entire axis, or
    - the sentinel `"..."` (or [`Ellipsis`][Ellipsis]), which may be used at
      most once and stands for all remaining axes, each of which is spanned
      entirely.

    For example, `chunk_shape=[1, "..."]` encodes every slice along the first
    axis independently, and `chunk_shape=[None, 100, 100]` encodes 100x100
    tiles that span the entire first axis.

    On encoding, the chunks are encoded from the first to the last axis (in
    row-major order) and the encoded chunks are framed together with the data
    type and shape information into a bytestring. On decoding, this framing
    information is used to decode each chunk into its place in the output
    array.

    Parameters
    ----------
    codec : dict | Codec
        The configuration or instantiated codec that encodes each chunk.
    chunk_shape : Sequence[None | int | str]
        The chunk shape, see above.
    """

    __slots__: tuple[str, ...] = ("_codec", "_chunk_shape")
    _codec: Codec
    _chunk_shape: tuple[None | int | str, ...]

    codec_id: str = "chunked"  # type: ignore

    def __init__(
        self,
        *,
        codec: dict | Codec,
        chunk_shape: Sequence[None | int | str | EllipsisType],
    ) -> None:
        self._codec = (
            codec if isinstance(codec, Codec) else numcodecs.registry.get_codec(codec)
        )

        normalised: list[None | int | str] = []
        for c in chunk_shape:
            if c is None:
                normalised.append(None)
            elif c is Ellipsis or c == ALL_OTHER_AXES:
                normalised.append(ALL_OTHER_AXES)
            elif isinstance(c, (int, np.integer)) and not isinstance(c, bool):
                if c <= 0:
                    raise ValueError("chunk lengths must be positive")
                normalised.append(int(c))
            else:
                raise TypeError(f"invalid chunk shape entry {c!r}")

        if normalised.count(ALL_OTHER_AXES) > 1:
            raise ValueError(f"the {ALL_OTHER_AXES!r} sentinel may only be used once")

        self._chunk_shape = tuple(normalised)

    def _resolve(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        """Resolve the chunk shape for an array of the given `shape`."""

        entries = list(self._chunk_shape)

        if ALL_OTHER_AXES in entries:
            i = entries.index(ALL_OTHER_AXES)
            n_other = len(shape) - (len(entries) - 1)
            if n_other < 0:
                raise ValueError(
                    f"chunk shape {self._chunk_shape} has more axes than the data shape {shape}"
                )
            entries[i : i + 1] = [None] * n_other
        elif len(entries) != len(shape):
            raise ValueError(
                f"chunk shape {self._chunk_shape} does not match the data shape {shape}"
            )

        return tuple(
            s if c is None else min(int(c), s) if s > 0 else 0  # type: ignore
            for c, s in zip(entries, shape)
        )

    @staticmethod
    def _chunks(shape: tuple[int, ...], chunk_shape: tuple[int, ...]):
        """Iterate over the slices of all chunks in row-major order."""

        ranges = [
            range(0, s, c) if c > 0 else range(0) for s, c in zip(shape, chunk_shape)
        ]
        for starts in itertools.product(*ranges):
            yield tuple(
                slice(start, min(start + c, s))
                for start, c, s in zip(starts, chunk_shape, shape)
            )

    def encode(self, buf: Buffer) -> bytes:
        """
        Encode the data in `buf`.

        Parameters
        ----------
        buf : Buffer
            Data to be encoded. May be any object supporting the new-style
            buffer protocol.

        Returns
        -------
        enc : bytes
            Encoded and framed chunks as a bytestring.
        """

        a = numcodecs.compat.ensure_ndarray(buf)
        dtype, shape = a.dtype, a.shape

        chunk_shape = self._resolve(shape)

        # message: dtype shape chunk-shape
        #          [encoded-dtype encoded-shape encoded-len encoded]*
        message: list[bytes | bytearray] = []

        message.append(leb128.u.encode(len(dtype.str)))
        message.append(dtype.str.encode("ascii"))

        message.append(leb128.u.encode(len(shape)))
        for s in shape:
            message.append(leb128.u.encode(s))

        for c in chunk_shape:
            message.append(leb128.u.encode(c))

        for slices in self._chunks(shape, chunk_shape):
            chunk = np.ascontiguousarray(a[slices])

            encoded = numcodecs.compat.ensure_ndarray(self._codec.encode(chunk))

            message.append(leb128.u.encode(len(encoded.dtype.str)))
            message.append(encoded.dtype.str.encode("ascii"))

            message.append(leb128.u.encode(encoded.ndim))
            for s in encoded.shape:
                message.append(leb128.u.encode(s))

            # ensure that the encoded values are encoded in little endian binary
            encoded_bytes = encoded.astype(encoded.dtype.newbyteorder("<")).tobytes()
            message.append(leb128.u.encode(len(encoded_bytes)))
            message.append(encoded_bytes)

        return b"".join(message)

    def decode(self, buf: Buffer, out: None | Buffer = None) -> Buffer:
        """
        Decode the data in `buf`.

        Parameters
        ----------
        buf : Buffer
            Encoded data. Must be an object representing a bytestring, e.g.
            [`bytes`][bytes] or a 1D array of [`np.uint8`][numpy.uint8]s etc.
        out : Buffer, optional
            Writeable buffer to store decoded data. N.B. if provided, this
            buffer must be exactly the right size to store the decoded data.

        Returns
        -------
        dec : Buffer
            Decoded data. May be any object supporting the new-style buffer
            protocol.
        """

        b = numcodecs.compat.ensure_bytes(buf)

        b_io = BytesIO(b)

        # message: dtype shape chunk-shape
        #          [encoded-dtype encoded-shape encoded-len encoded]*
        dtype = np.dtype(b_io.read(leb128.u.decode_reader(b_io)[0]).decode("ascii"))
        shape = tuple(
            leb128.u.decode_reader(b_io)[0]
            for _ in range(leb128.u.decode_reader(b_io)[0])
        )
        chunk_shape = tuple(leb128.u.decode_reader(b_io)[0] for _ in shape)

        decoded = np.empty(shape, dtype=dtype)

        for slices in self._chunks(shape, chunk_shape):
            encoded_dtype = np.dtype(
                b_io.read(leb128.u.decode_reader(b_io)[0]).decode("ascii")
            )
            encoded_shape = tuple(
                leb128.u.decode_reader(b_io)[0]
                for _ in range(leb128.u.decode_reader(b_io)[0])
            )
            encoded_len = leb128.u.decode_reader(b_io)[0]
            encoded_size = reduce(lambda a, b: a * b, encoded_shape, 1)

            encoded = (
                np.frombuffer(
                    b_io.read(encoded_len),
                    dtype=encoded_dtype.newbyteorder("<"),
                    count=encoded_size,
                )
                .astype(encoded_dtype)
                .reshape(encoded_shape)
            )

            chunk = np.empty(decoded[slices].shape, dtype=dtype)
            chunk_decoded = self._codec.decode(encoded, out=chunk)
            decoded[slices] = numcodecs.compat.ensure_ndarray(chunk_decoded).reshape(
                chunk.shape
            )

        return numcodecs.compat.ndarray_copy(decoded, out)  # type: ignore

    def get_config(self) -> dict:
        """
        Returns the configuration of this chunked meta-codec.

        [`numcodecs.registry.get_codec(config)`][numcodecs.registry.get_codec]
        can be used to reconstruct this codec from the returned config.

        Returns
        -------
        config : dict
            Configuration of this chunked meta-codec.
        """

        return dict(
            id=type(self).codec_id,
            codec=self._codec.get_config(),
            chunk_shape=list(self._chunk_shape),
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(codec={self._codec!r}, chunk_shape={list(self._chunk_shape)!r})"

    def map(self, mapper: Callable[[Codec], Codec]) -> "ChunkedCodec":
        """
        Apply the `mapper` to the inner `codec` of this chunked meta-codec.

        Parameters
        ----------
        mapper : Callable[[Codec], Codec]
            The callable that is applied to the inner codec.

        Returns
        -------
        mapped : ChunkedCodec
            The mapped chunked meta-codec.
        """

        return ChunkedCodec(codec=mapper(self._codec), chunk_shape=self._chunk_shape)


numcodecs.registry.register_codec(ChunkedCodec)
