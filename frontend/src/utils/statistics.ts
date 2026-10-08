/**
 * ============================================================
 *  肿瘤统计与报告生成工具
 * ============================================================
 * 本模块负责：
 *  1. calculateTumorStatistics —— 对分割标签逐体素统计各肿瘤区域（坏死核心/水肿区/增强肿瘤）的
 *     体素数、体积、含肿瘤切片范围等量化指标；
 *  2. formatVolume —— 把体积值格式化为带单位的中文友好字符串。
 *
 * 标签值约定（BraTS 分割标签惯例）：
 *   - 0：背景
 *   - 1：坏死核心（NCR，Necrotic core）
 *   - 2：水肿区（ED，Edema / peritumoral）
 *   - 3：增强肿瘤（ET，Enhancing tumor）
 *   - 4：在某些数据集中与 3 一同视为增强肿瘤区域
 */
import type { TumorStatistics, TypedArray } from '../types';
import { DEFAULT_VOLUME_DIMS } from '../config/appConfig';

/**
 * 计算肿瘤统计信息
 *
 * @param segData   分割标签的类型化数组（每个体素的值为 0~4 的标签）
 * @param mriHeader MRI 头部信息，用于读取维度 dims 与体素间距 pixDims
 * @returns TumorStatistics 完整的肿瘤量化统计结果
 *
 * 核心步骤：
 *  1. 遍历全部体素，按标签值归类计数（NCR=1、ED=2、ET=3或4）；
 *  2. 用 Set 收集所有含肿瘤体素的切片编号，避免重复；
 *  3. 由三个方向的体素间距相乘得到单个体素体积，乘以体素数得到各区域体积（mm³）；
 *  4. 汇总总体素、总体积、含肿瘤切片数与范围。
 */
export function calculateTumorStatistics(
  segData: TypedArray,
  mriHeader: { dims: number[]; pixDims: number[] }
): TumorStatistics {
  // 各区域体素计数器
  let necrosisCount = 0;      // 坏死核心（标签值 1）
  let edemaCount = 0;         // 水肿区（标签值 2）
  let enhancingCount = 0;     // 增强肿瘤（标签值 3 或 4）
  let totalTumorVoxels = 0;   // 肿瘤总体素数（三区域之和）

  // 用 Set 记录所有含肿瘤的切片编号，天然去重
  const tumorSlicesSet = new Set<number>();

  // 读取二维切片尺寸：轴位切片每层共 nx*ny 个体素
  const nx = mriHeader.dims[1] || DEFAULT_VOLUME_DIMS[1];
  const ny = mriHeader.dims[2] || DEFAULT_VOLUME_DIMS[2];
  const sliceSize = nx * ny;

  // 读取体素物理间距（mm），三者相乘即单个体素的体积
  const pixDimX = mriHeader.pixDims[1] || 1.0;
  const pixDimY = mriHeader.pixDims[2] || 1.0;
  const pixDimZ = mriHeader.pixDims[3] || 1.0;
  const voxelVolume = pixDimX * pixDimY * pixDimZ;

  // 逐体素扫描分割标签
  for (let i = 0; i < segData.length; i++) {
    // 标签值取整，兼容浮点存储的分割结果
    const val = Math.round(segData[i]);

    // 按 BraTS 标签值约定累加各区域体素
    if (val === 1) {
      necrosisCount++;
      totalTumorVoxels++;
    } else if (val === 2) {
      edemaCount++;
      totalTumorVoxels++;
    } else if (val === 3 || val === 4) {
      enhancingCount++;
      totalTumorVoxels++;
    }

    // 若该体素属于肿瘤（标签值 > 0），则把其所在切片编号加入集合
    if (val > 0) {
      // 切片编号 = 线性索引 / 单层体素数（向下取整）
      const sliceIndex = Math.floor(i / sliceSize);
      tumorSlicesSet.add(sliceIndex);
    }
  }

  // 含肿瘤切片总数
  const tumorSlices = tumorSlicesSet.size;
  // 把切片编号排序，便于求出范围 [min, max]
  const sliceNumbers = Array.from(tumorSlicesSet).sort((a, b) => a - b);
  const sliceRange: [number, number] = sliceNumbers.length > 0
    ? [sliceNumbers[0], sliceNumbers[sliceNumbers.length - 1]]
    : [0, 0];

  // 组装并返回统计结果；体积 = 体素数 × 单个体素体积（mm³）
  return {
    totalVoxels: totalTumorVoxels,
    necrosisVoxels: necrosisCount,
    edemaVoxels: edemaCount,
    enhancingVoxels: enhancingCount,
    necrosisVolume: necrosisCount * voxelVolume,
    edemaVolume: edemaCount * voxelVolume,
    enhancingVolume: enhancingCount * voxelVolume,
    totalVolume: totalTumorVoxels * voxelVolume,
    tumorSlices,
    sliceRange
  };
}

/**
 * 格式化体积值为可读字符串
 *
 * @param volumeMM3 体积值（单位 mm³）
 * @returns 当体积 ≥ 1000 mm³ 时换算为 cm³ 显示，否则保持 mm³，均保留两位小数
 */
export function formatVolume(volumeMM3: number): string {
  // 1 cm³ = 1000 mm³，体积较大时换算为 cm³ 更易读
  if (volumeMM3 >= 1000) {
    return `${(volumeMM3 / 1000).toFixed(2)} cm³`;
  }
  return `${volumeMM3.toFixed(2)} mm³`;
}

