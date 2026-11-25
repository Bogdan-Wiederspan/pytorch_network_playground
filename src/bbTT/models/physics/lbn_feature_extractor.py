from __future__ import annotations

import torch


class LBNFeatureExtractor(torch.nn.Module):
    """
    Helper Layer to filter out correct features necessary for the LBN to run. The filtering is done by naming!
    Thus changing names of features result automatically into  a broken functionality.

    Args:
        continuous_features (list[str]): List of all continuous features that should be extracted for the LBN.
    """

    def __init__(
        self,
        continuous_features: list[str],
        *args,
        **kwargs,
    ):
        super().__init__()
        self.continuous_features = continuous_features
        self.particles = self.find_particles_components_in_features()

    @property
    def num_particles(self):
        return len(self._particles)

    @property
    def _particles(self):
        return ("vis_tau1", "vis_tau2", "bjet1", "bjet2", "met", "nu1", "nu2")

    def find_particles_components_in_features(self):
        # returns dictionary with all indicies of the feature components
        positions = {name: idx for idx, name in enumerate(self.continuous_features)}

        def indicies(particle: str, components: tuple[str, ...]) -> list[int]:
            # exact names "<particle>_<component>": a substring search would also match derived
            # features such as "log_met_pt" for the particle "met"
            names = [f"{particle}_{component}" for component in components]
            missing = [name for name in names if name not in positions]
            if missing:
                raise ValueError(
                    f"the LBN needs the continuous features {missing}, add them to the feature list "
                    "or to DataConfig.lbn_features"
                )
            return [positions[name] for name in names]

        particles = {}

        for f in self._particles[:-3]:
            particles[f] = indicies(f, ("e", "px", "py", "pz"))
        particles["met"] = indicies("met", ("px", "py"))

        particles["nu1"] = indicies("nu1", ("px", "py", "pz"))
        particles["nu2"] = indicies("nu2", ("px", "py", "pz"))
        return particles

    def slice_particles_from_tensor(self, tensor):
        t = []
        for f in self._particles[:-3]:
            t.append(tensor[:, self.particles[f]])

        # met is special, since we need to reconstruct it: (pt, px, py ,0)
        met_kinematics = tensor[:, self.particles["met"]]

        met_pt = torch.sqrt(torch.sum(met_kinematics**2, axis=1))  # TODO float64?
        met_pz = torch.zeros_like(met_pt)
        met = torch.stack((met_pt, met_kinematics[:, 0], met_kinematics[:, 1], met_pz), axis=1)
        t.append(met)
        # add nu, separetley since pt = e: (pt, px, py, pz)
        for num in (1, 2):
            nu_kinematics = tensor[:, self.particles[f"nu{num}"]]
            nu_e = torch.sqrt(torch.sum(nu_kinematics**2, axis=1))
            nu = torch.stack((nu_e, nu_kinematics[:, 0], nu_kinematics[:, 1], nu_kinematics[:, 2]), axis=1)
            t.append(nu)
        # combine everything
        t = torch.stack(t, axis=-1)  # B, FEATURES (4), particles (7)
        return t

    def forward(self, x):
        return self.slice_particles_from_tensor(x)
        # return the indices of all particle features so the layer can slice them from tensors
