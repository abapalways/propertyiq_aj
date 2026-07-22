"""
Deterministic financial tools for PropertyIQ: mortgage calculation and 
comparable listings lookup. No LLM/embeddings involved — pure Python logic.
"""


def calc_mortgage(price, down_payment_pct, interest_rate, loan_term_years=30):
    """
    Calculate monthly mortgage payment (principal + interest only).

    Args:
        price: home price (float, > 0)
        down_payment_pct: down payment as a decimal, e.g. 0.20 for 20% (0-1)
        interest_rate: annual interest rate as a decimal, e.g. 0.065 for 6.5%
        loan_term_years: loan term in years (default 30)

    Returns:
        dict with monthly_payment, loan_amount, total_interest_paid

    Raises:
        ValueError: if any input is out of valid range
        TypeError: if a required input is missing/None
    """
    # 1. Validate inputs
    if price is None or down_payment_pct is None or interest_rate is None:
        raise TypeError("price, down_payment_pct, and interest_rate are required")

    if price <= 0:
        raise ValueError("price must be positive")

    if not (0 <= down_payment_pct <= 1):
        raise ValueError("down_payment_pct must be between 0 and 1")

    if not (0 <= interest_rate <= 0.30):
        raise ValueError("interest_rate must be between 0 and 0.30")

    if loan_term_years <= 0:
        raise ValueError("loan_term_years must be positive")

    # 2. Loan amount
    loan_amount = price - (price * down_payment_pct)

    # 3. Monthly rate and number of payments
    monthly_rate = interest_rate / 12
    num_payments = loan_term_years * 12

    # 4. Monthly payment formula
    if monthly_rate == 0:
        # Edge case: 0% interest, formula would divide by zero
        monthly_payment = loan_amount / num_payments
    else:
        monthly_payment = loan_amount * (
            monthly_rate * (1 + monthly_rate) ** num_payments
        ) / (
            (1 + monthly_rate) ** num_payments - 1
        )

    # 5. Total interest paid
    total_paid = monthly_payment * num_payments
    total_interest_paid = total_paid - loan_amount

    # 6. Return
    return {
        "monthly_payment": round(monthly_payment, 2),
        "loan_amount": round(loan_amount, 2),
        "total_interest_paid": round(total_interest_paid, 2),
    }

if __name__ == "__main__":
    result = calc_mortgage(price=425000, down_payment_pct=0.20, interest_rate=0.065)
    print(result)

    # Error case tests
    try:
        calc_mortgage(price=-100, down_payment_pct=0.2, interest_rate=0.065)
    except ValueError as e:
        print(f"Correctly caught: {e}")

    try:
        calc_mortgage(price=425000, down_payment_pct=1.5, interest_rate=0.065)
    except ValueError as e:
        print(f"Correctly caught: {e}")

    try:
        calc_mortgage(price=425000, down_payment_pct=0.20, interest_rate=None)
    except TypeError as e:
        print(f"Correctly caught: {e}")

    # 0% interest edge case
    result_zero = calc_mortgage(price=300000, down_payment_pct=0.20, interest_rate=0.0)
    print(f"0% interest case: {result_zero}")