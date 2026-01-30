from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, Set, List, Union
from copy import deepcopy
import torch

from boltz.data import const
from boltz.model.potentials.schedules import *

class Potential(ABC):
    def __init__(
        self,
        parameters: Optional[Dict[str, Union[ParameterSchedule, float, int, bool]]] = None
    ):
        self.parameters = parameters

    def compute(self, coords, feats, parameters):
        index, args, com_args = self.compute_args(feats, parameters)

        if index.shape[1] == 0:
            return torch.zeros(coords.shape[:-2], device=coords.device)

        #print("COMPUTING POTENTIAL")
        if com_args is not None:
            #print("WITH COM")
            # If using COM, define COM groups, along with atom padding mask for valid atoms
            # then compute COM of each unpadded group of atom coords
            # each group then has an index, this index is used to compute variable
            # e.g. to compute the distance between COMs of two groups
            com_index, atom_pad_mask = com_args
            unpad_com_index = com_index[atom_pad_mask]
            unpad_coords = coords[...,atom_pad_mask,:]
            coords = torch.zeros(
                (*unpad_coords.shape[:-2], unpad_com_index.max() + 1, 3),
                device=coords.device
            ).scatter_reduce(
                -2,
                unpad_com_index.unsqueeze(-1).expand_as(unpad_coords),
                unpad_coords,
                'mean'
            )
        value = self.compute_variable(coords, index, compute_gradient=False)
        energy = self.compute_function(value, *args)
        return energy.sum(dim=-1)

    def compute_gradient(self, coords, feats, parameters):
        index, args, com_args = self.compute_args(feats, parameters)
        #print("COMPUTING GRADIENT")
        if com_args is not None:
            com_index, atom_pad_mask = com_args
        else:
            com_index, atom_pad_mask = None, None

        # TODO: can this be moved up and above bool consolidated with below?
        if index.shape[1] == 0:
            return torch.zeros_like(coords)

        if com_index is not None:
            #print("WITH COM")
            unpad_coords = coords[...,atom_pad_mask,:]
            unpad_com_index = com_index[atom_pad_mask]
            coords = torch.zeros(
                (*unpad_coords.shape[:-2], unpad_com_index.max() + 1, 3),
                device=coords.device
            ).scatter_reduce(
                -2,
                unpad_com_index.unsqueeze(-1).expand_as(unpad_coords),
                unpad_coords,
                'mean'
            )
            com_counts = torch.bincount(com_index[atom_pad_mask])

        value, grad_value = self.compute_variable(coords, index, compute_gradient=True)
        energy, dEnergy = self.compute_function(value, *args, compute_derivative=True)

        grad_atom = torch.zeros_like(coords).scatter_reduce(
            -2,
            index.flatten(start_dim=0, end_dim=1).unsqueeze(-1).expand((*coords.shape[:-2],-1,3)),
            dEnergy.tile(grad_value.shape[-3]).unsqueeze(-1) * grad_value.flatten(start_dim=-3, end_dim=-2),
            'sum',
        )

        if com_index is not None:
            grad_atom = grad_atom[..., com_index, :]

        return grad_atom

    def compute_parameters(self, t):
        if self.parameters is None:
            return None
        parameters = {
            name: parameter if not isinstance(parameter, ParameterSchedule) else parameter.compute(t)
            for name, parameter in self.parameters.items()
        }
        return parameters

    @abstractmethod
    def compute_function(self, value, *args, compute_derivative=False):
        raise NotImplementedError

    @abstractmethod
    def compute_variable(self, coords, index, compute_gradient=False):
        raise NotImplementedError

    @abstractmethod
    def compute_args(self, t, feats, **parameters):
        raise NotImplementedError

class FlatBottomPotential(Potential):
    def compute_function(self, value, k, lower_bounds, upper_bounds, compute_derivative=False):
        if lower_bounds is None:
            lower_bounds = torch.full_like(value, float('-inf'))
        if upper_bounds is None:
            upper_bounds = torch.full_like(value, float('inf'))

        neg_overflow_mask = value < lower_bounds
        pos_overflow_mask = value > upper_bounds

        energy = torch.zeros_like(value)
        energy[neg_overflow_mask] = (k * (lower_bounds - value))[neg_overflow_mask]
        energy[pos_overflow_mask] = (k * (value - upper_bounds))[pos_overflow_mask]
        if not compute_derivative:
            return energy

        dEnergy = torch.zeros_like(value)
        dEnergy[neg_overflow_mask] = -1 * k.expand_as(neg_overflow_mask)[neg_overflow_mask]
        dEnergy[pos_overflow_mask] = 1 * k.expand_as(pos_overflow_mask)[pos_overflow_mask]

        return energy, dEnergy
        
class DistancePotential(Potential):
    def compute_variable(self, coords, index, compute_gradient=False):
        r_ij = coords.index_select(-2, index[0]) -  coords.index_select(-2, index[1])
        r_ij_norm = torch.linalg.norm(r_ij, dim=-1)
        r_hat_ij = r_ij / r_ij_norm.unsqueeze(-1)

        if not compute_gradient:
            return r_ij_norm

        grad_i = r_hat_ij
        grad_j = -1 * r_hat_ij
        grad = torch.stack((grad_i, grad_j), dim=1)

        return r_ij_norm, grad

class DihedralPotential(Potential):
    def compute_variable(self, coords, index, compute_gradient=False):
        r_ij = coords.index_select(-2, index[0]) - coords.index_select(-2, index[1])
        r_kj = coords.index_select(-2, index[2]) - coords.index_select(-2, index[1])
        r_kl = coords.index_select(-2, index[2]) - coords.index_select(-2, index[3])
    
        n_ijk = torch.cross(r_ij, r_kj, dim=-1)
        n_jkl = torch.cross(r_kj, r_kl, dim=-1)
        
        r_kj_norm = torch.linalg.norm(r_kj, dim=-1)
        n_ijk_norm = torch.linalg.norm(n_ijk, dim=-1)
        n_jkl_norm = torch.linalg.norm(n_jkl, dim=-1)
        
        sign_phi = torch.sign(r_kj.unsqueeze(-2) @ torch.cross(n_ijk, n_jkl, dim=-1).unsqueeze(-1)).squeeze(-1, -2)
        phi = sign_phi * torch.arccos(
            torch.clamp(
                (n_ijk.unsqueeze(-2) @ n_jkl.unsqueeze(-1)).squeeze(-1,-2) /
                (n_ijk_norm * n_jkl_norm),
                -1 + 1e-8,
                1 - 1e-8
            )
        )

        if not compute_gradient:
            return phi

        a = ((r_ij.unsqueeze(-2) @ r_kj.unsqueeze(-1)).squeeze(-1, -2) / (r_kj_norm ** 2)).unsqueeze(-1)
        b = ((r_kl.unsqueeze(-2) @ r_kj.unsqueeze(-1)).squeeze(-1, -2) / (r_kj_norm ** 2)).unsqueeze(-1)
        
        grad_i = n_ijk * (r_kj_norm / n_ijk_norm ** 2).unsqueeze(-1)
        grad_l = -1 * n_jkl * (r_kj_norm / n_jkl_norm ** 2).unsqueeze(-1)
        grad_j = (a - 1) * grad_i - b * grad_l
        grad_k = (b - 1) * grad_l - a * grad_i
        grad = torch.stack((grad_i, grad_j, grad_k, grad_l), dim=1)        
        return phi, grad

class AbsDihedralPotential(DihedralPotential):
    def compute_variable(self, coords, index, compute_gradient=False):
        if not compute_gradient:
            phi = super().compute_variable(coords, index, compute_gradient=compute_gradient)
            phi = torch.abs(phi)
            return phi

        phi, grad = super().compute_variable(coords, index, compute_gradient=compute_gradient)
        grad[(phi < 0)[...,None,:,None].expand_as(grad)] *= -1
        phi = torch.abs(phi)

        return phi, grad 

class PoseBustersPotential(FlatBottomPotential, DistancePotential):
    def compute_args(self, feats, parameters):
        pair_index = feats['rdkit_bounds_index'][0]
        lower_bounds = feats['rdkit_lower_bounds'][0].clone()
        upper_bounds = feats['rdkit_upper_bounds'][0].clone()
        bond_mask = feats['rdkit_bounds_bond_mask'][0]
        angle_mask = feats['rdkit_bounds_angle_mask'][0]
        
        lower_bounds[bond_mask * ~angle_mask] *= (1.0 - parameters['bond_buffer'])
        upper_bounds[bond_mask * ~angle_mask] *= (1.0 + parameters['bond_buffer'])
        lower_bounds[~bond_mask * angle_mask] *= (1.0 - parameters['angle_buffer'])
        upper_bounds[~bond_mask * angle_mask] *= (1.0 + parameters['angle_buffer'])
        lower_bounds[bond_mask * angle_mask] *= (1.0 - min(parameters['angle_buffer'], parameters['angle_buffer']))
        upper_bounds[bond_mask * angle_mask] *= (1.0 + min(parameters['angle_buffer'], parameters['angle_buffer']))
        lower_bounds[~bond_mask * ~angle_mask] *= (1.0 - parameters['clash_buffer'])
        upper_bounds[~bond_mask * ~angle_mask] = float('inf')

        k = torch.ones_like(lower_bounds)

        return pair_index, (k, lower_bounds, upper_bounds), None

class ConnectionsPotential(FlatBottomPotential, DistancePotential):
    def compute_args(self, feats, parameters):
        pair_index = feats['connected_atom_index'][0]
        lower_bounds = None
        upper_bounds = torch.full((pair_index.shape[1],), parameters['buffer'], device=pair_index.device)
        k = torch.ones_like(upper_bounds)

        return pair_index, (k, lower_bounds, upper_bounds), None
        
class VDWOverlapPotential(FlatBottomPotential, DistancePotential):
    def compute_args(self, feats, parameters):
        atom_chain_id = (
            torch.bmm(feats["atom_to_token"].float(), feats["asym_id"].unsqueeze(-1).float())
            .squeeze(-1)
            .long()
        )[0]
        atom_pad_mask = feats['atom_pad_mask'][0].bool()
        chain_sizes = torch.bincount(atom_chain_id[atom_pad_mask])
        single_ion_mask = (chain_sizes > 1)[atom_chain_id]

        vdw_radii = torch.zeros(const.num_elements, dtype=torch.float32, device=atom_chain_id.device)
        vdw_radii[1:119] = torch.tensor(const.vdw_radii, dtype=torch.float32, device=atom_chain_id.device)
        atom_vdw_radii = (feats['ref_element'].float() @ vdw_radii.unsqueeze(-1)).squeeze(-1)[0]

        pair_index = torch.triu_indices(atom_chain_id.shape[0], atom_chain_id.shape[0], 1, device=atom_chain_id.device)
        
        pair_pad_mask = atom_pad_mask[pair_index].all(dim=0)       
        pair_ion_mask = single_ion_mask[pair_index[0]] * single_ion_mask[pair_index[1]]

        num_chains = atom_chain_id.max() + 1
        connected_chain_index = feats['connected_chain_index'][0]
        connected_chain_matrix = torch.eye(num_chains, device=atom_chain_id.device, dtype=torch.bool)
        connected_chain_matrix[connected_chain_index[0], connected_chain_index[1]] = True
        connected_chain_matrix[connected_chain_index[1], connected_chain_index[0]] = True
        connected_chain_mask = connected_chain_matrix[atom_chain_id[pair_index[0]], atom_chain_id[pair_index[1]]]

        pair_index = pair_index[:, pair_pad_mask * pair_ion_mask * ~connected_chain_mask]

        lower_bounds = atom_vdw_radii[pair_index].sum(dim=0) * (1.0 - parameters['buffer'])
        upper_bounds = None
        k = torch.ones_like(lower_bounds)

        return pair_index, (k, lower_bounds, upper_bounds), None


class SymmetricChainCOMPotential(FlatBottomPotential, DistancePotential):
    def compute_args(self, feats, parameters):
        atom_chain_id = (
            torch.bmm(feats["atom_to_token"].float(), feats["asym_id"].unsqueeze(-1).float())
            .squeeze(-1)
            .long()
        )[0]
        atom_pad_mask = feats['atom_pad_mask'][0].bool()
        chain_sizes = torch.bincount(atom_chain_id[atom_pad_mask])
        single_ion_mask = chain_sizes > 1

        pair_index = feats['symmetric_chain_index'][0]
        pair_ion_mask = single_ion_mask[pair_index[0]] * single_ion_mask[pair_index[1]]
        pair_index = pair_index[:, pair_ion_mask]
        lower_bounds = torch.full((pair_index.shape[1],), parameters['buffer'], dtype=torch.float32, device=pair_index.device)
        upper_bounds = None
        k = torch.ones_like(lower_bounds)
    
        return pair_index, (k, lower_bounds, upper_bounds), (atom_chain_id, atom_pad_mask)

class StereoBondPotential(FlatBottomPotential, AbsDihedralPotential):
    def compute_args(self, feats, parameters):
        stereo_bond_index = feats['stereo_bond_index'][0]
        stereo_bond_orientations = feats['stereo_bond_orientations'][0].bool()
        
        lower_bounds = torch.zeros(stereo_bond_orientations.shape, device=stereo_bond_orientations.device)
        upper_bounds = torch.zeros(stereo_bond_orientations.shape, device=stereo_bond_orientations.device)
        lower_bounds[stereo_bond_orientations] = torch.pi - parameters['buffer']
        upper_bounds[stereo_bond_orientations] = float('inf')
        lower_bounds[~stereo_bond_orientations] = float('-inf')
        upper_bounds[~stereo_bond_orientations] = parameters['buffer']
        
        k = torch.ones_like(lower_bounds)

        return stereo_bond_index, (k, lower_bounds, upper_bounds), None

class ChiralAtomPotential(FlatBottomPotential, DihedralPotential):
    def compute_args(self, feats, parameters):
        chiral_atom_index = feats['chiral_atom_index'][0]
        chiral_atom_orientations = feats['chiral_atom_orientations'][0].bool()

        lower_bounds = torch.zeros(chiral_atom_orientations.shape, device=chiral_atom_orientations.device)
        upper_bounds = torch.zeros(chiral_atom_orientations.shape, device=chiral_atom_orientations.device)
        lower_bounds[chiral_atom_orientations] = parameters['buffer']
        upper_bounds[chiral_atom_orientations] = float('inf')
        upper_bounds[~chiral_atom_orientations] = -1 * parameters['buffer']
        lower_bounds[~chiral_atom_orientations] = float('-inf')

        k = torch.ones_like(lower_bounds)
        return chiral_atom_index, (k, lower_bounds, upper_bounds), None

class PlanarBondPotential(FlatBottomPotential, AbsDihedralPotential):
    def compute_args(self, feats, parameters):
        double_bond_index = feats['planar_bond_index'][0].T
        double_bond_improper_index = torch.tensor([
            [1,2,3,0],
            [4,5,0,3],
        ], device=double_bond_index.device).T
        improper_index = double_bond_index[:, double_bond_improper_index].swapaxes(0,1).flatten(start_dim=1)
        lower_bounds = None
        upper_bounds = torch.full((improper_index.shape[1],), parameters['buffer'], device=improper_index.device)
        k = torch.ones_like(upper_bounds)

        return improper_index, (k, lower_bounds, upper_bounds), None

# test a single contact distance potential
class ContactPotential(FlatBottomPotential, DistancePotential):
    def compute_args(self, feats, parameters):
        # TODO: for com_args: return com_group_ids and atom_pad_mask
        #       then return the index for the com_groups that are being calculated
        #index = feats["contact_pair_index"][0]

        # assign two groups: ligand and receptor
        # TODO: eventually include these in feats for generality
        lig_indices = [1382, 1383, 1384, 1385, 1386, 1387]
        rec_indices = [650, 768, 773, 789, 797, 856, 904]

        # assign com group id of 0 to other, 1 to receptor, 2 to ligand
        # start with zero tensor of shape (num_atoms,)
        atom_group_id = torch.zeros(feats["atom_pad_mask"].shape[1], dtype=torch.long, device=feats["atom_pad_mask"].device)
        atom_group_id[rec_indices] = 1
        atom_group_id[lig_indices] = 2
        #print(f"{atom_group_id=}: {atom_group_id=}")

        # TODO: prob need a mask here when selecting the contact pair
        # note that index is for coords, which is of shape (n_fk_particles, n_padded_atoms, 3)
        # so need to return indices that are valid for the padded atoms only
        # but the padding is just for the last few atoms, that is it padds the ends of the tensor dim 
        atom_pad_mask = feats['atom_pad_mask'][0].bool()
        #print(f"{atom_pad_mask.shape=}")
        # TODO: dummy test index, use same device as feats, e.g. GPU
        #index = torch.tensor([[5],[20]], device=feats["atom_pad_mask"].device)
        # 1 is receptor group and 2 is ligand group
        index = torch.tensor([[1],[2]], device=feats["atom_pad_mask"].device)

        # Angstrom : fixed distance for testing (TODO: also pull from feats)
        distance = 10
        lower_bounds = torch.full((index.shape[1],), distance - parameters['buffer'], device=index.device)
        upper_bounds = torch.full((index.shape[1],), distance + parameters['buffer'], device=index.device)
        # force constant: 1.0 for each contact
        k = torch.ones_like(lower_bounds)

        #return index, (k, lower_bounds, upper_bounds), None
        return index, (k, lower_bounds, upper_bounds), (atom_group_id, atom_pad_mask)

def get_potentials():
    potentials = [
        # SymmetricChainCOMPotential(
        #     parameters={
        #         'guidance_interval': 4,
        #         'guidance_weight': 0.5,
        #         'resampling_weight': 0.5,
        #         'buffer': ExponentialInterpolation(
        #             start=1.0,
        #             end=5.0,
        #             alpha=-2.0
        #         )
        #     }
        # ),
        # VDWOverlapPotential(
        #     parameters={
        #         'guidance_interval': 5,
        #         'guidance_weight': PiecewiseStepFunction(
        #             thresholds=[0.4],
        #             values=[0.125, 0.0]
        #         ),
        #         'resampling_weight': PiecewiseStepFunction(
        #             thresholds=[0.6],
        #             values=[0.01, 0.0]
        #         ),
        #         'buffer': 0.225,
        #     }
        # ),
        # ConnectionsPotential(
        #     parameters={
        #         'guidance_interval': 1,
        #         'guidance_weight': 0.15,
        #         'resampling_weight': 1.0,
        #         'buffer': 2.0,
        #     }
        # ),
        # PoseBustersPotential(
        #     parameters={
        #         'guidance_interval': 1,
        #         'guidance_weight': 0.05,
        #         'resampling_weight': 0.1,
        #         'bond_buffer': 0.20,
        #         'angle_buffer': 0.20,
        #         'clash_buffer': 0.15
        #     }
        # ),
        # ChiralAtomPotential(
        #     parameters={
        #         'guidance_interval': 1,
        #         'guidance_weight': 0.10,
        #         'resampling_weight': 1.0,
        #         'buffer': 0.52360
        #     }
        # ),
        # StereoBondPotential(
        #     parameters={
        #         'guidance_interval': 1,
        #         'guidance_weight': 0.05,
        #         'resampling_weight': 1.0,
        #         'buffer': 0.52360
        #     }
        # ),
        # PlanarBondPotential(
        #     parameters={
        #         'guidance_interval': 1,
        #         'guidance_weight': 0.05,
        #         'resampling_weight': 1.0,
        #         'buffer': 0.26180
        #     }
        # ),
        ContactPotential(
            parameters={
                'guidance_interval': 1,
                'guidance_weight': 0.5,
                'resampling_weight': 1.0,
                'buffer': 1.0, # e.g. here a distance buffer, but normally the actual distance
            }
        ),
    ]
    return potentials
