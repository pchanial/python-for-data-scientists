def oof_model(freq, param):
    """
    1/f noise model
        param[0] : noise standard deviation
        param[1] : knee frequency
        param[2] : alpha
    freq : array of frequency
    """
    sigma, fknee, alpha = param
    return sigma**2 * (1 + (fknee / freq)**alpha)


def compute_residuals(param, observation, freq):
    """
    Return array: observation - model
    """
    model = oof_model(freq, param)
    residual = np.log(observation / model)
    print("residual: ", np.sum(residual**2))
    return residual

# fit with scipy optimize, leastsq() function
param_true = np.array([np.std(signal), 1, 1.2])
param_guess = param_true * np.random.uniform(0.5, 2, 3)
ret_lsq = spo.leastsq(compute_residuals, param_guess, args=(spectrum, freq),
                      full_output=True)
