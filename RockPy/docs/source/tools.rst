Tools
*****

This is the collection of tools, that are user-firendly.

pandas_tools
============

.. automodule:: RockPy.tools.pandas_tools
    :members:

compute
=======

rotations
---------
.. automodule:: RockPy.tools.compute
    :members: rotmat, rx, ry, rz, rotate, rotate_around_axis, rotate_arbitrary

conversion
----------

Conversion functions transform data from one coordinate system into a different one. The shape of the input will
be handled by the decorator `RockPy.core.utils.handle_shape_dtype`.

See Also:
    :py:func:`RockPy.core.utils.handle_shape_dtype`

.. automodule:: RockPy.tools.compute
    :members: convert_to_xyz, convert_to_dim, convert_to_stereographic, convert_to_equal_area, convert_to_hvl
