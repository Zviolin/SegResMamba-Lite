/**
 * 后处理单元测试：softmax / argmax / 类别统计 / 逆变换回填
 * 校验逻辑与 www 原版输出一致（医学项目防 AI 改坏的保险）。
 */
import { describe, it, expect } from 'vitest';
import {
  softmax4,
  processMultiClassOutput,
  countClasses,
  upsampleMaskToOriginal,
} from '../postprocess';
import { TARGET_SIZE } from '../preprocess';

const classSize = TARGET_SIZE * TARGET_SIZE * TARGET_SIZE;

describe('softmax4（www softmax4 对齐）', () => {
  it('概率和为 1，且最大值对应 argmax 类别', () => {
    const input = new Float32Array([1, 2, 3, 4]);
    const p = softmax4(input, 0, 1);
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6);
    expect(p[3]).toBeGreaterThan(p[2]);
    expect(p[2]).toBeGreaterThan(p[1]);
    expect(p[1]).toBeGreaterThan(p[0]);
  });
});

describe('processMultiClassOutput（argmax）', () => {
  it('全背景 logits 最高 → 标签全 0', () => {
    const out = new Float32Array(4 * classSize);
    out.fill(0.5);
    for (let i = 0; i < classSize; i++) out[i] = 2.0; // 类别 0 更高
    const labels = processMultiClassOutput(out);
    expect(labels.every((v) => v === 0)).toBe(true);
  });

  it('单个 voxel 类别 3 最高 → 该位置标签 3，其余 0', () => {
    const out = new Float32Array(4 * classSize);
    out.fill(0.5);
    for (let i = 0; i < classSize; i++) out[i] = 2.0; // 默认类别 0 高
    const v = 12345;
    out[3 * classSize + v] = 9.0; // 类别 3 最大
    const labels = processMultiClassOutput(out);
    expect(labels[v]).toBe(3);
    expect(labels[v - 1]).toBe(0);
    expect(labels[v + 1]).toBe(0);
  });
});

describe('countClasses', () => {
  it('统计各类别体素数', () => {
    const labels = new Uint8Array(100);
    for (let i = 0; i < 100; i++) labels[i] = i % 4;
    const counts = countClasses(labels);
    expect(counts[0]).toBe(25);
    expect(counts[1]).toBe(25);
    expect(counts[2]).toBe(25);
    expect(counts[3]).toBe(25);
  });
});

describe('upsampleMaskToOriginal（逆变换回填，历史 bug 回归测试）', () => {
  it('64³ 标签经 逆重采样→逆裁剪→逆填充 回到原始正确位置', () => {
    // 模拟变换：原始 8×8×4 → 重采样 [4,4,2] → crop bbox [1,1,0,2,2,1]（cropShape 2×2×2）→ pad 64³ start [31,31,31]
    const transform = {
      resampledShape: [4, 4, 2] as [number, number, number],
      cropBBox: [1, 1, 0, 2, 2, 1] as [number, number, number, number, number, number],
      cropShape: [2, 2, 2] as [number, number, number],
      padStarts: [31, 31, 31] as [number, number, number],
    };
    // 64³ 中 (d=31,h=31,w=31) 对应 crop 坐标 (0,0,0)，标签置 2
    const labels = new Uint8Array(classSize);
    labels[31 * TARGET_SIZE * TARGET_SIZE + 31 * TARGET_SIZE + 31] = 2;

    const seg = upsampleMaskToOriginal(labels, 8, 8, 4, transform);

    // 逆映射（与函数公式完全一致）：
    //   resX = round(x*(resW-1)/max(1,nx-1)) = round(x*1/7)   → resX=0 时 x∈{0,1,2,3}
    //   resY = round(y*(resH-1)/max(1,ny-1)) = round(y*3/7)   → resY=1 时 y∈{2,3}
    //   resZ = round(z*(resD-1)/max(1,nz-1)) = round(z*3/3)   → resZ=1 时 z=1
    // 标签 2 位于 64³ (d=31,h=31,w=31)：w=resX+31 → resX=0；h=resY-1+31 → resY=1；d=resZ-1+31 → resZ=1
    let found = 0;
    for (let z = 0; z < 4; z++) {
      const resZ = Math.round((z * (4 - 1)) / Math.max(1, 4 - 1));
      for (let y = 0; y < 8; y++) {
        const resY = Math.round((y * (4 - 1)) / Math.max(1, 8 - 1));
        for (let x = 0; x < 8; x++) {
          const resX = Math.round((x * (2 - 1)) / Math.max(1, 8 - 1));
          const expectLabel =
            resZ === 1 && resY === 1 && resX === 0 ? 2 : 0;
          const val = seg[x + y * 8 + z * 64];
          expect(val).toBe(expectLabel);
          if (expectLabel === 2) found++;
        }
      }
    }
    expect(found).toBe(8); // x∈{0..3} × y∈{2,3} × z=1
  });

  it('无变换参数时回退纯比例缩放（不崩溃）', () => {
    const labels = new Uint8Array(classSize);
    labels[0] = 1;
    const seg = upsampleMaskToOriginal(labels, 8, 8, 4, {
      resampledShape: null,
      cropBBox: null,
      cropShape: null,
      padStarts: [0, 0, 0],
    });
    expect(seg.length).toBe(8 * 8 * 4);
  });
});
