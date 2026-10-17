from cer.eval.significance import paired_permutation_test


def test_identical_systems():
    a = {f"q{i}": i / 10 for i in range(20)}
    assert paired_permutation_test(a, a)["p_value"] == 1.0


def test_clear_difference_is_significant():
    a = {f"q{i}": 0.8 for i in range(30)}
    b = {f"q{i}": 0.5 + (i % 3) * 0.01 for i in range(30)}
    r = paired_permutation_test(a, b, n_permutations=2000)
    assert r["delta"] > 0.25 and r["p_value"] < 0.01
