from src.model_utils import energy_direct_cost


def test_bs_cost_scales_with_energy():
    assert energy_direct_cost(0.45, 4000) == 1800


def test_sp_and_ae_costs_scale_with_energy():
    assert energy_direct_cost(0.15, 4000) == 600
    assert energy_direct_cost(0.90, 4000) == 3600
