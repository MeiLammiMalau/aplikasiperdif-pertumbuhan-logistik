import numpy as np
from scipy.optimize import curve_fit

def logistic_model(t, N0, r, K):
    """
    Solusi analitik Model Pertumbuhan Logistik.
    N(t) = K / (1 + (K/N0 - 1) * exp(-r*t))
    """
    if N0 <= 0 or K <= 0:
        return np.inf

    val = (K / N0 - 1) * np.exp(-r * t)
    return K / (1 + val)

def estimate_logistic_params(t_data, N_data, initial_guesses):
    """
    Estimasi parameter logistik (N0, r, K) dari data waktu dan populasi.
    Parameter awal disediakan oleh pengguna.
    """
    try:
        popt, pcov = curve_fit(logistic_model, t_data, N_data, p0=initial_guesses, maxfev=10000)

        if popt[0] <= 0 or popt[1] <= 0 or popt[2] <= 0:
            raise ValueError("Parameter yang diestimasi tidak valid (negatif atau nol).")

        return popt, pcov
    except RuntimeError as e:
        raise ValueError(f"Gagal mengestimasi parameter: {e}")
    except Exception as e:
        raise Exception(f"Terjadi kesalahan tak terduga saat estimasi: {e}")

def estimate_logistic_params_auto(t_data, N_data):
    """
    Estimasi otomatis parameter logistik (N0, r, K) tanpa input manual tebakan awal.
    """
    # Estimasi awal yang sederhana dan umum
    N0_guess = N_data[0]
    K_guess = max(N_data) * 1.5  # diasumsikan populasi belum mencapai K
    r_guess = 0.1  # tebakan konservatif untuk laju pertumbuhan

    initial_guesses = [N0_guess, r_guess, K_guess]
    return estimate_logistic_params(t_data, N_data, initial_guesses)
