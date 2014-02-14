
.. _broadcasting_shapes:

Can the arrays of the following shapes be broadcast together? If yes, what would be the shape of the result?

* (7, 1) and (7, 4) Yes (7, 4)

* (7,) and (4, 7) Yes (4, 7)

* (3, 3) and (2, 3) No: the first axes are incompatible

* (1, 1, 1, 8) and (9, 1) Yes (1, 1, 9, 8)

* (4, 1, 9) and (3, 1) Yes (4, 3, 9)
