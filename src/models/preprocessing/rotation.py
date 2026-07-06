from __future__ import annotations

import torch


class RotatePhiLayer(torch.nn.Module):  # noqa: F811
    def __init__(
        self,
        columns: list[str] | None,
        ref_phi_columns: list[str] | None = ("lepton1", "lepton2"),
        rotate_columns: list[str] | None = ("bjet1", "bjet2", "fatjet", "lepton1", "lepton2"),
        separator="_",
        active: bool = True,
    ):
        """
        Rotate specific *columns* given in *rotate_columns* relative to reference in *ref_phi_columns*.
        TODO proper docstring
        """
        super().__init__()
        self.separator=separator
        self.ref_indices = torch.nn.Buffer(self.find_indices_of(columns, ref_phi_columns, True), persistent=True)
        self.rotate_indices = torch.nn.Buffer(self.find_indices_of(columns, rotate_columns, True), persistent=True)
        self.active = active

    def load_state_dict(self, state_dict: dict, strict: bool, assign: bool):
        # overload load_state_dict to set buffer sizes to same of state dict
        for name in self.state_dict().keys():
            self.__setattr__(name, torch.zeros_like(state_dict[name]))
        super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign)

    def find_indices_of(
        self,
        search_in: list[str],
        search_for: list[str],
        _expand: bool = False,

    ) -> torch.FloatTensor | None:
        if search_in is None or search_for is None:
            return None

        if _expand:
            search_for = self._expand(search_for, separator=self.separator)
        return torch.tensor([tuple(map(search_in.index, particle)) for particle in search_for])

    def _expand(self, columns: list[str] | str, separator="_") -> list[tuple[str, str]]:
        # adds px, py to columns and return them as tuple
        columns = [columns] if isinstance(columns, str) else columns
        return [tuple(f"{col}{separator}{suffix}" for suffix in ("px", "py")) for col in columns]

    def calc_phi(
        self,
        x: torch.FloatTensor,
        y: torch.FloatTensor,
    ) -> torch.FloatTensor:
        def arctan2(
            x: torch.FloatTensor,
            y: torch.FloatTensor,
        ) -> torch.FloatTensor:
            # calculate arctan2 using arctan, since onnx does not have support for arctan2
            # torch constants are not tensors for some reason
            pi = torch.tensor(torch.pi)

            # handle special cases
            # quadrant handling: (x < 0,y >= 0) -> arctan + pi, (x < 0,y < 0) -> arctan - pi
            phi_shift = torch.zeros_like(x)
            torch.where(torch.logical_and(x < 0, y < 0), -pi, phi_shift, out=phi_shift)
            torch.where(torch.logical_and(x < 0, y >= 0), pi, phi_shift, out=phi_shift)

            # edges  with x == 0
            phis = torch.arctan(y / x)
            # right edge -> pi/2, left edge -> -pi/2, both 0 -> 0
            torch.where(torch.logical_and(x == 0, y > 0), (pi / 2), phis, out=phis)
            torch.where(torch.logical_and(x == 0, y < 0), -(pi / 2), phis, out=phis)
            torch.where(torch.logical_and(x == 0, y == 0), torch.tensor(0), phis, out=phis)
            return phis + phi_shift
        return arctan2(x=x, y=y)

    def rotate_pt_to_phi(
        self,
        px: torch.FloatTensor,
        py: torch.FloatTensor,
        ref_phi: torch.FloatTensor,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        # rotate px, py relative to ref_phi
        # returns px, py rotated
        pt = torch.sqrt(torch.square(px) + torch.square(py))
        old_phi = self.calc_phi(x=px, y=py)
        new_phi = old_phi - ref_phi
        return pt * torch.cos(new_phi), pt * torch.sin(new_phi)

    def calc_ref_phi(
        self,
        array,
    ) -> torch.FloatTensor:
        px1, py1 = self.get_kinematics(array, self.ref_indices[0])
        px2, py2 = self.get_kinematics(array, self.ref_indices[1])

        ref_phi = self.calc_phi(
            x=px1 + px2,
            y=py1 + py2,
        )
        return ref_phi

    def get_kinematics(
        self,
        array: torch.FloatTensor,
        ref_indices: torch.FloatTensor,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        x, y = ref_indices
        return array[:, x], array[:, y]

    def rotate_columns(self, array: torch.FloatTensor, ref_phi: torch.FloatTensor) -> torch.FloatTensor:
        # px, py pairs in fixed order
        for rotate_indice in self.rotate_indices:
            px, py = self.get_kinematics(array, rotate_indice)

            new_px, new_py = self.rotate_pt_to_phi(
                px=px,
                py=py,
                ref_phi=ref_phi,
            )
            array[:, rotate_indice[0]] = new_px
            array[:, rotate_indice[1]] = new_py
        return array

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        if not self.active:
            return x
        ref_phi = self.calc_ref_phi(x)
        return self.rotate_columns(x, ref_phi)
