from src.solvers.cg_solver import _positive_congestion_prices, _reduced_cost


def test_positive_congestion_makes_high_congestion_less_attractive():
    duals = {"shore": {(0, 5): -1000.0, (0, 6): -1.0}, "battery": {}, "ship": {}}
    rho, eta = _positive_congestion_prices(duals)
    base = {"cost": 100.0, "battery_use": []}
    high = dict(base, shore_berth_use=[(0, 5)])
    low = dict(base, shore_berth_use=[(0, 6)])

    assert _reduced_cost(low, pi_i=0.0, rho=rho, eta=eta) < _reduced_cost(high, pi_i=0.0, rho=rho, eta=eta)
