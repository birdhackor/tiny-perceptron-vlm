Original HTTPS URL: https://raw.githubusercontent.com/python/cpython/v3.13.5/Doc/library/time.rst
Original version: CPython v3.13.5
Original lines: 321-348
.. function:: perf_counter() -> float

   .. index::
      single: benchmarking

   Return the value (in fractional seconds) of a performance counter, i.e. a
   clock with the highest available resolution to measure a short duration.  It
   does include time elapsed during sleep and is system-wide.  The reference
   point of the returned value is undefined, so that only the difference between
   the results of two calls is valid.

   .. impl-detail::

      On CPython, use the same clock as :func:`time.monotonic` and is a
      monotonic clock, i.e. a clock that cannot go backwards.

   Use :func:`perf_counter_ns` to avoid the precision loss caused by the
   :class:`float` type.

   .. versionadded:: 3.3

   .. versionchanged:: 3.10
      On Windows, the function is now system-wide.

   .. versionchanged:: 3.13
      Use the same clock as :func:`time.monotonic`.


