"""V6 vs V2 vs SegMamba-Official 完整对比"""
import pandas as pd

v_lite = pd.read_csv(r'g:/Codes/Python/SRTP/Essay/PythonFiles/Code/SegResMamba-Lite/pipeline/evaluation_results/metrics_segresmamba_lite_2.0mm.csv')
v_lite_lw = pd.read_csv(r'g:/Codes/Python/SRTP/Essay/PythonFiles/Code/SegResMamba-Lite/pipeline/evaluation_results/metrics_segresmamba_lite_2.0mm_lesion.csv', usecols=['CaseID','LW_Dice_WT','LW_Dice_TC','LW_Dice_ET','LW_HD95_WT','LW_HD95_TC','LW_HD95_ET'])
v_off = pd.read_csv(r'g:/Codes/Python/SRTP/Essay/PythonFiles/Code/SegResMamba-Lite/pipeline/evaluation_results/metrics_segresmamba_2.0mm.csv')
v_off_lw = pd.read_csv(r'g:/Codes/Python/SRTP/Essay/PythonFiles/Code/SegResMamba-Lite/pipeline/evaluation_results/metrics_segresmamba_2.0mm_lesion.csv', usecols=['CaseID','LW_Dice_WT','LW_Dice_TC','LW_Dice_ET','LW_HD95_WT','LW_HD95_TC','LW_HD95_ET'])

print('=' * 80)
print('全局指标对比')
print('=' * 80)
header = "版本".ljust(28) + "Dice_WT".rjust(10) + "Dice_TC".rjust(10) + "Dice_ET".rjust(10) + "HD95_WT".rjust(10) + "HD95_TC".rjust(10) + "HD95_ET".rjust(10)
print(header)
print('-' * 80)

for name, df in [('SegResMamba-Lite (V2/V6)', v_lite), ('SegMamba-Official (30M)', v_off)]:
    row = name.ljust(28)
    for c in ['Dice_WT', 'Dice_TC', 'Dice_ET']:
        row += f'{df[c].mean():.4f}'.rjust(10)
    for c in ['HD95_WT', 'HD95_TC', 'HD95_ET']:
        row += f'{df[c].mean():.4f}'.rjust(10)
    print(row)

print()
print('=' * 80)
print('Lesion-wise 指标对比')
print('=' * 80)
header = "版本".ljust(28) + "LWD_WT".rjust(10) + "LWD_TC".rjust(10) + "LWD_ET".rjust(10) + "LHW_WT".rjust(10) + "LHW_TC".rjust(10) + "LHW_ET".rjust(10)
print(header)
print('-' * 80)

for name, df in [('SegResMamba-Lite (V2/V6)', v_lite_lw), ('SegMamba-Official (30M)', v_off_lw)]:
    row = name.ljust(28)
    for c in ['LW_Dice_WT', 'LW_Dice_TC', 'LW_Dice_ET']:
        row += f'{df[c].mean():.4f}'.rjust(10)
    for c in ['LW_HD95_WT', 'LW_HD95_TC', 'LW_HD95_ET']:
        row += f'{df[c].mean():.4f}'.rjust(10)
    print(row)
