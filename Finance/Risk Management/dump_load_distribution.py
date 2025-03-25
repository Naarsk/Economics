import numpy as np
from scipy.stats import rv_continuous
from scipy.interpolate import interp1d


def dump_data(x_vals, pdf_vals, filename="my_distribution"):
    if len(x_vals) != len(pdf_vals):
        raise ValueError("x_vals and pdf_vals must have the same length.")
    np.savez(filename+".npz", x_vals=x_vals, pdf_vals=pdf_vals)
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

        # Optionally, pre-compute CDF by cumulative integration
        # to implement _cdf() or _ppf(). For demonstration, we skip that.

    def _pdf(self, x):
        """
        Return the PDF at points x by interpolation.
        """
        return self.pdf_interpolator(x)

    # If you want a working .rvs(), .cdf(), etc. you can also implement:
    #   def _cdf(self, x):
    #       ...
    #   def _ppf(self, q):
    #       ...
    #   def _stats(self):
    #       ...
    # but they are optional if you only need .pdf() or sampling.

