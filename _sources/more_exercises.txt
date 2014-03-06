More exercises
--------------

.. topic:: **Exercise**: Power spectrum & data fitting
    :class: green

    1. Download the **FITS file** :download:`data_oof.fits <auto_examples/data_oof.fits>`, read and run the following example of power spectrum estimate:

    .. literalinclude:: auto_examples/plot_oof1.py

    2. Complete the ellipses in the following code, to estimate the 1/f noise parameters of the signal.

    .. literalinclude:: auto_examples/plot_oof2.py

    .. only:: html

        [:ref:`Solution <plot_oof_solution.py>`]


.. topic:: **Exercise**: Condition number & error propagation.
    :class: green

    Given

    .. math:: A &= \frac{1}{4}
                   \begin{bmatrix}
                       3 & -\sqrt{3} & 1 & -\sqrt{3} \\
                       \sqrt{3} & 3 & -\sqrt{3} & -1 \\
                       \sqrt{3} & 1 &  \sqrt{3} &  3 \\
                       1 & -\sqrt{3} & -3 & \sqrt{3} \\
                   \end{bmatrix},\quad
              B =  \begin{bmatrix}
                       10 & 10 &  7 & 8 \\
                        9 &  2 &  7 & 7 \\
                        1 &  5 & 11 & 1 \\
                       10 & 11 &  4 & 8 \\
                   \end{bmatrix}, \quad
              x =  \begin{bmatrix}
                       1\\
                       1\\
                       1\\
                       1\\
                   \end{bmatrix} \\
              b_A &= A x \\
              b_B &= B x \\

    1. Check that :math:`A` is orthogonal.
    2. Compute the condition numbers of :math:`A` and :math:`B`.
    3. Compute :math:`A^{-1}` and :math:`B^{-1}`.
    4. By Monte-Carlo sampling, check how perturbations in :math:`b_A` and :math:`b_B` are propagated to the solutions

       .. math:: x + \Delta x_A &= A^{-1}(b_A + \Delta b_A) \\
                 x + \Delta x_B &= B^{-1}(b_B + \Delta b_B) \\

       by computing the mean ratio of the relative errors :math:`\frac{||\Delta x||}{||x||}` and :math:`\frac{||\Delta b||}{||b||}` for :math:`A` and :math:`B`. Compare these values to the condition numbers of the matrices.

    .. only:: html

        [:ref:`Solution <condnum.py>`]


.. topic:: **Exercise**: Scientific constants & integration
    :class: green

    Numerically verify the Stephan-Boltzmann law, which relates the emission of a black body to its temperature.

