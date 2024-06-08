import torch
import torch.nn as nn

from model.nnloader import *


# HMTL model
class HeteroNN(nn.Module):
    def __init__(self, md_files:dict) -> None:
        super(HeteroNN, self).__init__()
        assert all(md_type in md_files.keys() for md_type in ['proj_encoder', 'fus_encoder', 'indep_solver'])
        assert set(md_files['proj_encoder'].keys()) == set(md_files['indep_solver'].keys())

        self.pj_encoder = {}
        self.fu_encoder = load_nn(md_files['fus_encoder'])
        self.id_solver = {}

        for md, file_path in md_files['proj_encoder'].items():
            self.pj_encoder[md] = load_nn(file_path)
        
        for md, file_path in md_files['indep_solver'].items():
            self.id_solver[md] = load_nn(file_path)

    def forward_train(self, ds:dict) -> tuple:
        y = dict(); z = dict()

        for ds_name, input_data in ds.items():
            x_bar = self.pj_encoder[ds_name](input_data)
            z_ = self.fu_encoder(x_bar)

            y[ds_name] = self.id_solver[ds_name](z_)
            z[ds_name] = z_

        return y, z

    def forward_test(self, ds_name:str, input_data:torch.tensor) -> tuple:
        x_bar = self.pj_encoder[ds_name](input_data)
        z_ = self.fu_encoder(x_bar)

        y = self.id_solver[ds_name](z_)

        return y, z_

    # Overriding
    def cuda(self) -> None:
        for pe_md, is_md in zip(self.pj_encoder.values(), self.id_solver.values()):
            pe_md = pe_md.cuda()
            is_md = is_md.cuda()

        self.fu_encoder = self.fu_encoder.cuda()

    # Overriding
    def parameters(self) -> list:
        params = list(self.fu_encoder.parameters())

        for pe_md, is_md in zip(self.pj_encoder.values(), self.id_solver.values()):
            params += list(pe_md.parameters())
            params += list(is_md.parameters())

        return params

    # Overriding
    def train(self) -> None:
        self.fu_encoder.train()

        for pe_md, is_md in zip(self.pj_encoder.values(), self.id_solver.values()):
            pe_md.train()
            is_md.train()