import pandas as pd

from source_code.adversaries.kpp_attack import KppAttack
from source_code.adversaries.mk_attack import MkAttack
from source_code.adversaries.stat_attack import StatAttack
from source_code.adversaries.hyp_attack import HypVolAttack
from pathlib import Path
import os
import sys

root_path = Path(__file__).parent.parent.parent
data_path = os.path.join(root_path, "processed_data\\dsn_keystroke")
bootstrap_data_path = os.path.join(data_path, 'bs_data')
cluster_data_path = os.path.join(data_path, 'cluster_data')
hyp_vol_data_path = os.path.join(data_path, 'hyper_volume_data')

feature_paths = {f'gr{gr}': os.path.join(data_path, f'df_group_{gr}_gr_scl.csv') for gr in range(1, 3)}


data = {f'{gr}': pd.read_csv(path) for gr, path in feature_paths.items()}

num_samples = 100

kpp_adv = KppAttack(data=data['gr2'], required_attack_samples=num_samples)
kpp_adv_data = kpp_adv.generate_attack()

mk_adv = MkAttack(data=data['gr2'], required_attack_samples=num_samples)
mk_adv_data = mk_adv.generate_attack()

stat_adv = StatAttack(data=data['gr2'], required_attack_samples=num_samples, bootstrap_data_path=bootstrap_data_path,
                      run_bootstrap=True, bootstrap_iter=100)
stat_adv_data = stat_adv.generate_attack()

num_cls = 6
rand_state = 42
hyp_adv =  HypVolAttack(data=data['gr2'], equal_user_data=False, random_state=rand_state, calc_clusters=False,
                               clusters_path=cluster_data_path, gr_num=1, cluster_count=num_cls,
                               ol_path=hyp_vol_data_path, attack_samples=num_samples,
                               ol_cut_off=0, std_dev_at_gr=None)
hyp_adv_data = hyp_adv.generate_attack()
a = 1