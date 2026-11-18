"""Export target-independent H4 inputs for combination with team models."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from ..features.epu import load_epu_index, align_epu_to_base_month, calculate_h4_epu_features

OUTPUT = 'data/processed/h4_epu.parquet'

def extra_feature_specs():
    # Keep monthly EPU variation and the interaction scale: do not rank these.
    return {name: {'path': OUTPUT, 'column': column, 'transform': 'none'}
            for name, column in [('h4_mom','mom'),('h4_epu','epu_raw'),('h4_interaction','h4_interaction')]}

def export(root=Path.cwd()):
    root=Path(root).resolve()
    mom=pd.read_parquet(root/'data/processed/mom.parquet')
    mom['base_month']=pd.PeriodIndex(mom.base_month,freq='M').astype(str)
    mom=mom[mom.base_month.between('2000-12','2021-11')].copy()
    mom['mom']=mom['mom'].where(mom.mom_available.eq(True))
    keys=mom.loc[mom.eligible.eq(True),['stock_id','base_month']]
    epu=align_epu_to_base_month(load_epu_index(root/'data/raw/Korea_Overall_EPU.xlsx'))
    result=calculate_h4_epu_features(mom,keys,epu)
    result=result.sort_values(['base_month','stock_id']).reset_index(drop=True)
    if not result.stock_id.astype(str).str.fullmatch(r'[A-Za-z0-9]{6}').all():
        raise ValueError('invalid stock identifier')
    if np.isinf(result[['mom','epu_raw','h4_interaction']].to_numpy(float)).any():
        raise ValueError('nonfinite model features')
    path=root/OUTPUT
    path.parent.mkdir(parents=True,exist_ok=True)
    result.to_parquet(path,index=False)
    manifest=dict(definition='Cho-Kim (2023) Korean EPU(t-1), team raw 11-calendar-month MOM(t), product',source_page='https://policyuncertainty.com/korea_chokim.html',
                  local_sheet='Korean New EPU Index',source_verified='local workbook attribution cites Cho and Kim (2023); download date and release vintages unverified',
                  target_independent=True,raw_rows=len(result),available_rows=int(result.h4_available.sum()),extra_features=extra_feature_specs(),pit_verified=False,model_integration_tested=False,
                  hashes={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in ['data/processed/mom.parquet','data/raw/Korea_Overall_EPU.xlsx',OUTPUT]})
    folder=root/'outputs/run_id/h4_input_export'
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'model_input_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    return manifest

if __name__=='__main__':
    print(json.dumps(export(),ensure_ascii=False,indent=2))
