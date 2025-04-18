import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import interp1d
from scipy.stats import rv_continuous
import os

def dump_distribution(x_vals, pdf_vals, cdf_vals=None, filename="my_distribution"):
    if len(x_vals) != len(pdf_vals):
        raise ValueError("x_vals and pdf_vals must have the same length.")
    if cdf_vals:
        np.savez(filename+".npz", x_vals=x_vals, pdf_vals=pdf_vals, cdf_vals=cdf_vals)
    else:
        np.savez(filename+".npz", x_vals=x_vals, pdf_vals=pdf_vals)
    cwd = os.getcwd()
    print("saved to:", cwd+"/"+filename+".npz")
    return

class CustomDistribution(rv_continuous):
    def __init__(self, data_file, *args, **kwargs):
        """
        data_file: Path to an .npz file containing x_vals and pdf_vals.
        """
        super().__init__(*args, **kwargs)

        # Load the numerical PDF from disk, ensuring they are floats
        data = np.load(data_file)
        self.x_vals = np.asarray(data["x_vals"], dtype=float)
        self.pdf_vals = np.asarray(data["pdf_vals"], dtype=float)

        # Build an interpolator for the PDF
        self.pdf_interpolator = interp1d(
            self.x_vals,
            self.pdf_vals,
            bounds_error=False,
            fill_value=0.0,  # PDF=0 outside the range
            kind='linear'
        )

        # Compute the CDF if not provided
        if 'cdf_vals' in data:
            self.cdf_vals = np.asarray(data["cdf_vals"], dtype=float)
        else:
            # Compute CDF via Simpson's rule over the PDF values
            cdf_partial = [simpson(self.pdf_vals[:i], self.x_vals[:i]) for i in range(1, len(self.x_vals))]
            # Prepend 0.0 (CDF at the lower bound is 0)
            self.cdf_vals = np.concatenate(([0.0], cdf_partial))
            # Normalize to ensure the CDF ends at 1
            self.cdf_vals /= self.cdf_vals[-1]

        # Build an interpolator for the CDF
        self.cdf_interpolator = interp1d(
            self.x_vals,
            self.cdf_vals,
            bounds_error=False,
            fill_value=(0.0, 1.0),  # Ensures proper boundary values for CDF
            kind='linear'
        )

        # Set the support boundaries as floats for the built-in inversion routines
        self.a = float(self.x_vals[0])
        self.b = float(self.x_vals[-1])

    def _pdf(self, x, *args):
        """
        Return the PDF at points x by interpolation.
        """
        # Ensure x is a float array to avoid casting issues
        x = np.asarray(x, dtype=float)
        return self.pdf_interpolator(x)

    def _cdf(self, x, *args):
        """
        Return the CDF at points x by interpolation.
        """
        # Ensure x is a float array
        x = np.asarray(x, dtype=float)
        return self.cdf_interpolator(x)


