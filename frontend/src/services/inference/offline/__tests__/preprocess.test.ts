/**
 * 预处理单元测试：重采样 / 裁剪 / 填充 / 归一化 / 模态映射
 * 校验逻辑与 www 原版一致（医学项目防 AI 改坏的保险）。
 */
import { describe, it, expect } from 'vitest';
import {
  resizeVolume,
  padToSize,
  cropForeground,
  normalizeVolume,
  mapModalityToChannel,
} from '../preprocess';

describe('resizeVolume（三线性插值，www:2671）', () => {
  it('2×2×2 → 3×3×3，中点等于 8 角均值', () => {
    const data = new Float32Array([0, 10, 20, 30, 40, 50, 60, 70]);
    const out = resizeVolume(data, [2, 2, 2], [3, 3, 3]);
    expect(out.length).toBe(27);
    // 中点 (d=1,h=1,w=1)：oldDf=oldHf=oldWf=1 → 8 角均值 35
    expect(out[1 * 9 + 1 * 3 + 1]).toBeCloseTo(35, 5);
    // 角落 (0,0,0) = 原值 0
    expect(out[0]).toBeCloseTo(0, 5);
  });
});

describe('padToSize（中心填充，www:2524）', () => {
  it('2×2×2 → 4×4×4：中心为 1，边缘为 0', () => {
    const data = new Float32Array(8).fill(1);
    const out = padToSize(data, [2, 2, 2], [4, 4, 4]);
    expect(out.length).toBe(64);
    expect(out[1 * 16 + 1 * 4 + 1]).toBe(1); // 中心
    expect(out[0]).toBe(0);                  // 角落背景
    expect(out[63]).toBe(0);
  });
});

describe('cropForeground（前景裁剪，www:2489）', () => {
  it('4×4×4 仅中心 2×2×2 非零 → 裁剪为 2×2×2，bbox 正确', () => {
    const data = new Float32Array(64);
    for (let d = 1; d < 3; d++)
      for (let h = 1; h < 3; h++)
        for (let w = 1; w < 3; w++)
          data[d * 16 + h * 4 + w] = 5;
    const { data: cropped, shape, bbox } = cropForeground(data, [4, 4, 4]);
    expect(shape).toEqual([2, 2, 2]);
    expect(bbox).toEqual([1, 1, 1, 2, 2, 2]);
    expect(cropped.length).toBe(8);
    expect(cropped[0]).toBe(5);
  });

  it('全零数据 → 返回 64³ 空体（与 www 一致）', () => {
    const data = new Float32Array(64);
    const { shape, data: cropped } = cropForeground(data, [4, 4, 4]);
    expect(shape).toEqual([64, 64, 64]);
    expect(cropped.every((v) => v === 0)).toBe(true);
  });
});

describe('normalizeVolume（1-99 百分位，www:2627）', () => {
  it('1..1000 数据：最大值映射 ≈1，背景 0 保持 0', () => {
    const data = new Float32Array(1000);
    for (let i = 0; i < 1000; i++) data[i] = i + 1;
    const out = normalizeVolume(data);
    expect(out.length).toBe(1000);
    expect(out[999]).toBeCloseTo(1, 3); // 最高值 → 1
    expect(out[0]).toBeCloseTo(0, 3);   // p01≈11 → 0
  });

  it('含背景 0 时背景位置保持 0', () => {
    const data = new Float32Array(20);
    data.fill(0);
    for (let i = 5; i < 20; i++) data[i] = i * 10;
    const out = normalizeVolume(data);
    expect(out[0]).toBe(0);
    expect(out[19]).toBeCloseTo(1, 4);
  });
});

describe('mapModalityToChannel（在线模态 → 离线通道）', () => {
  it('标准映射正确', () => {
    expect(mapModalityToChannel('T1ce')).toBe('t1c');
    expect(mapModalityToChannel('FLAIR')).toBe('t2f');
    expect(mapModalityToChannel('T1')).toBe('t1n');
    expect(mapModalityToChannel('T2')).toBe('t2w');
    expect(mapModalityToChannel('unknown')).toBe('t1c'); // 默认
  });
});
