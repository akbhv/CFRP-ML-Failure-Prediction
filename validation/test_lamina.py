from src.micromechanics import rule_of_mixtures_E1

Ef = 230e9
Em = 3.5e9
Vf = 0.60

E1 = rule_of_mixtures_E1(Ef, Em, Vf)

print(f"E1 = {E1 / 1e9:.2f} GPa")
