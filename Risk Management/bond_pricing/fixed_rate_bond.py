import pandas as pd
import numpy as np
from scipy.optimize import newton


def make_schedule(first_coupon_date, maturity_date, frequency):
    payment_day = pd.Timestamp(first_coupon_date).day
    schedule = pd.date_range(start=first_coupon_date, end=maturity_date,
                             freq=frequency) + pd.DateOffset(days=payment_day - 1)
    return schedule

def make_cashflows(schedule : pd.DatetimeIndex, yearly_rate, nominal, num_payments_per_year):
    num_payments = len(schedule)
    coupon_rate = yearly_rate * num_payments_per_year / 12
    cashflows = np.zeros(num_payments)

    for i in range(num_payments):
        cashflows[i] = nominal * coupon_rate

    cashflows[-1] += nominal

    return pd.DataFrame({'Cashflows': cashflows}, index=schedule)

def discount_cashflows(bond : pd.DataFrame, zero_rates):
    num_payments = len(bond)
    discounted_bond = bond
    cashflows = discounted_bond['Cashflows'].to_numpy()

    discount_factors = np.zeros(num_payments)

    for i in range(num_payments):
        discount_factors[i] = 1/ (1 + zero_rates[i])**i

    discounted_cashflows = cashflows * discount_factors

    discounted_bond["Discount Factors"] = discount_factors
    discounted_bond["Discounted Cashflows"] = discounted_cashflows

    return discounted_bond

def duration(bond, zero_rates):
    ds = discount_cashflows(bond, zero_rates)["Discounted Cashflows"]
    return sum(ds*(1+np.arange(len(bond))))/sum(ds)

def price(bond, zero_rates):
    ds = discount_cashflows(bond, zero_rates)["Discounted Cashflows"]
    return sum(ds)

def convexity(bond, zero_rates):
    ds = discount_cashflows(bond, zero_rates)["Discounted Cashflows"]
    return sum(ds*(1+np.arange(len(bond)))*(2+np.arange(len(bond))))/sum(ds)

def yield_to_maturity(bond, zero_rates):

    cashflows = bond['Cashflows'].to_numpy()

    target_price = price(bond, zero_rates)

    # Function to solve for YTM
    def ytm_func(y):
        return np.sum([cashflows[t] / (1 + y) ** t for t in range(len(cashflows))]) - target_price

    # Solve for ytm
    ytm = newton(ytm_func, x0=0.025)

    return ytm