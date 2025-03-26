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

        # Load the numerical PDF from disk
        data = np.load(data_file)
        self.x_vals = data["x_vals"]
        self.pdf_vals = data["pdf_vals"]

        # Build an interpolator for the PDF
        # kind='linear' or 'cubic' depending on how smooth your PDF is
        self.pdf_interpolator = interp1d(
            self.x_vals,
            self.pdf_vals,
            bounds_error=False,
            fill_value=0.0,  # PDF=0 outside the range
            kind='linear'
        )

        if 'cdf_vals' in data:
            self.cdf_vals =  data["cdf_vals"]
        else:
            self.cdf_vals = np.concat([np.array([self.pdf_vals[0]]), np.array([simpson(self.pdf_vals[:i], self.x_vals[:i]) for i in range(1,len(self.x_vals))])])

        self.cdf_interpolator = interp1d(
            self.x_vals,
            self.cdf_vals,
            bounds_error=False,
            fill_value=(0.0,1.0),
            kind='linear'
        )

    def _pdf(self, x, *args):
        """
        Return the PDF at points x by interpolation.
        """
        return self.pdf_interpolator(x)

    def _cdf(self, x, *args):
        """
        Return the CDF at points x by interpolation.
        """
        return self.cdf_interpolator(x)



