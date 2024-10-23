import numpy as np
import pandas as pd


class ExogenousVariables:
    def __init__(self, labor : np.array, hicks_progress : np.array, solow_progress : np.array, harrod_progress : np.array):
        """
        Initialize an ExogenousVariables object.

        Parameters
        ----------
        labor : array
            Exogenous labor levels.
        hicks_progress : array
            Exogenous Hicks-neutral technological progress.
        solow_progress : array
            Exogenous Solow-neutral technological progress.
        harrod_progress : array
            Exogenous Harrod-neutral technological progress.
        """
        self.labor = labor
        self.hicks_progress = hicks_progress
        self.solow_progress = solow_progress
        self.harrod_progress = harrod_progress

    def __call__(self, time_horizon=None):
        if time_horizon is None:
            time_horizon = len(self.labor)
        elif type(time_horizon) is not int:
            raise Exception('time_horizon must be an integer')
        elif time_horizon > len(self.labor):
            raise Exception('time_horizon must be less than or equal to len(labor)')
        else:
            self.labor = self.labor[0:time_horizon]
        return pd.DataFrame({'time': range(time_horizon), 'labor': self.labor, 'hicks_progress': self.hicks_progress,
                             'solow_progress': self.solow_progress, 'harrods_progress': self.harrod_progress})


class ExogenousFunctions(ExogenousVariables):
    def __init__(self, starting_pop, starting_hicks, starting_solow, starting_harrods, labor_growth_function,
                 hicks_progress_function, solow_progress_function, harrods_progress_function, time_horizon=1000):
        """
        Initialize an ExogenousFunctions object.

        Parameters
        ----------
        starting_pop : float
            Initial population level.
        starting_hicks : float
            Initial Hicks-neutral technological progress.
        starting_solow : float
            Initial Solow-neutral technological progress.
        starting_harrods : float
            Initial Harrod-neutral technological progress.
        labor_growth_function : callable
            Function to generate labor growth.
        hicks_progress_function : callable
            Function to generate Hicks-neutral technological progress.
        solow_progress_function : callable
            Function to generate Solow-neutral technological progress.
        harrods_progress_function : callable
            Function to generate Harrod-neutral technological progress.
        time_horizon : int, optional
            Number of periods to generate. Defaults to 1000.

        Returns
        -------
        None
        """
        time_horizon = time_horizon
        labor = np.zeros(time_horizon)
        labor[0] = starting_pop
        hicks_progress = np.zeros(time_horizon)
        hicks_progress[0] = starting_hicks
        solow_progress = np.zeros(time_horizon)
        solow_progress[0] = starting_solow
        harrod_progress = np.zeros(time_horizon)
        harrod_progress[0] = starting_harrods

        for t in range(1, time_horizon):
            labor[t] = labor_growth_function(labor[t - 1])
            hicks_progress[t] = hicks_progress_function(hicks_progress[t - 1])
            solow_progress[t] = solow_progress_function(solow_progress[t - 1])
            harrod_progress[t] = harrods_progress_function(harrod_progress[t - 1])

        super().__init__(labor, hicks_progress, solow_progress, harrod_progress)
