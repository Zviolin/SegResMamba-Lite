/** LABEL_COLORS： BraTS 三个肿瘤区域的唯一颜色/标签值来源，视图叠加、图例、勾选控件都必须从这里取色。 */
export const LABEL_COLORS = {
  NCR: { label: '坏死核心', short: 'NCR', value: 1, key: 'showNecrosis', hex: '#64B5F6', rgb: [100, 181, 246] },
  ED:  { label: '水肿区',   short: 'ED',  value: 2, key: 'showEdema',    hex: '#81C784', rgb: [129, 199, 132] },
  ET:  { label: '增强肿瘤', short: 'ET',  values: [3, 4], key: 'showEnhancing', hex: '#E57373', rgb: [229, 115, 115] },
} as const;

/** 图例展示顺序（ET → ED → NCR，与移动端/历史详情页的紧凑图例一致） */
export const LEGEND_ORDER = [LABEL_COLORS.ET, LABEL_COLORS.ED, LABEL_COLORS.NCR] as const;
