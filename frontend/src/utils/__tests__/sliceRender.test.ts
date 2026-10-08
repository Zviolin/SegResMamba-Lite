/**
 * 切片渲染链路单元测试：getSliceData 越界 / applyWindowLevel 退化 / renderSlicePixels 形状不符
 *
 * 背景：这三个函数处在「上传文件 → 屏幕成像」的关键路径上，
 * 历史上都出过「不报错但画面全黑或叠加错位」的静默失败，故在此锁定行为。
 */
import { describe, it, expect } from 'vitest';
import { getSliceData, applyWindowLevel, getVolumeDims, getMaxSlices } from '../niftiParser';
import { renderSlicePixels } from '../blend';
import type { ViewSettings, TypedArray, NiftiData } from '../../types';

/** 构造一组可反推坐标的三维体数据：idx = z*nx*ny + y*nx + x，值为 z*1000 + y*10 + x */
function makeVolume(nx: number, ny: number, nz: number): TypedArray {
  const data = new Float32Array(nx * ny * nz);
  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        data[z * nx * ny + y * nx + x] = z * 1000 + y * 10 + x;
      }
    }
  }
  return data;
}

describe('getSliceData —— 形状与坐标', () => {
  const dims = [3, 4, 5, 6]; // nx=4, ny=5, nz=6
  const vol = makeVolume(4, 5, 6);

  it('轴位切片：宽= X、高= Y，像素按 Z 层取值', () => {
    const { pixels, width, height } = getSliceData(vol, dims, 2, 'axial');
    expect(width).toBe(4);
    expect(height).toBe(5);
    expect(pixels.length).toBe(4 * 5);
    // 第一个像素对应 (x=0, y=0, z=2)
    expect(pixels[0]).toBe(2 * 1000);
  });

  it('冠状位切片：宽= X、高= Z，首行取自顶层(z=nz-1)', () => {
    const { pixels, width, height } = getSliceData(vol, dims, 3, 'coronal');
    expect(width).toBe(4);
    expect(height).toBe(6);
    // 第一行是 z=5，固定 y=3
    expect(pixels[0]).toBe(5 * 1000 + 3 * 10);
  });

  it('矢状位切片：宽= Y、高= Z，首像素取自顶层且 Y 翻转', () => {
    const { pixels, width, height } = getSliceData(vol, dims, 1, 'sagittal');
    expect(width).toBe(5);
    expect(height).toBe(6);
    // 第一行是 z=5 且 y=ny-1=4，固定 x=1
    expect(pixels[0]).toBe(5 * 1000 + 4 * 10 + 1);
  });
});

describe('getSliceData —— 切片索引越界保护', () => {
  const dims = [3, 4, 5, 6];
  const vol = makeVolume(4, 5, 6);

  it('超过 Z 上界时夹到最后一帧，不产生 undefined', () => {
    const { pixels } = getSliceData(vol, dims, 999, 'axial');
    expect(pixels.length).toBe(20);
    // 应等价于 index = nz-1 = 5
    expect(pixels[0]).toBe(5 * 1000);
    expect(pixels.every((v) => typeof v === 'number' && !Number.isNaN(v))).toBe(true);
  });

  it('负索引夹到 0，不产生 undefined', () => {
    const { pixels } = getSliceData(vol, dims, -5, 'axial');
    expect(pixels[0]).toBe(0);
    expect(pixels.every((v) => typeof v === 'number' && !Number.isNaN(v))).toBe(true);
  });

  it('冠状位 Y 越界同样被夹住', () => {
    const { pixels } = getSliceData(vol, dims, 99, 'coronal');
    // y 应为 ny-1 = 4
    expect(pixels[0]).toBe(5 * 1000 + 4 * 10);
    expect(pixels.every((v) => typeof v === 'number' && !Number.isNaN(v))).toBe(true);
  });

  it('矢状位 X 越界同样被夹住', () => {
    const { pixels } = getSliceData(vol, dims, 99, 'sagittal');
    // x 应为 nx-1 = 3
    expect(pixels[0]).toBe(5 * 1000 + 4 * 10 + 3);
    expect(pixels.every((v) => typeof v === 'number' && !Number.isNaN(v))).toBe(true);
  });
});

describe('applyWindowLevel —— 退化参数不产生 NaN', () => {
  it('窗宽为 0 时不产生 NaN（否则整屏静默全黑）', () => {
    const out = applyWindowLevel(500, 0, 400);
    expect(Number.isNaN(out)).toBe(false);
    expect(out).toBeGreaterThanOrEqual(0);
    expect(out).toBeLessThanOrEqual(255);
  });

  it('窗宽为负数时不产生 NaN', () => {
    expect(Number.isNaN(applyWindowLevel(500, -100, 400))).toBe(false);
  });

  it('窗宽 / 窗位为 NaN 时仍能返回合法灰度', () => {
    const out = applyWindowLevel(500, Number.NaN, Number.NaN);
    expect(Number.isNaN(out)).toBe(false);
    expect(out).toBeGreaterThanOrEqual(0);
    expect(out).toBeLessThanOrEqual(255);
  });

  it('正常区间：区间外的值被钳到 0 / 255', () => {
    // 窗位 100、窗宽 20 → 区间 [90, 110]
    expect(applyWindowLevel(90, 20, 100)).toBe(0);
    expect(applyWindowLevel(110, 20, 100)).toBe(255);
    expect(applyWindowLevel(100, 20, 100)).toBeCloseTo(127.5, 5);
  });
});

describe('renderSlicePixels —— 标签与灰度长度不符时不叠加', () => {
  const viewSettings: ViewSettings = {
    windowWidth: 800,
    windowLevel: 400,
    opacity: 100,
    showNecrosis: true,
    showEdema: true,
    showEnhancing: true,
  };

  /** 造一个只带 data 字段的 ImageData（node 环境没有 DOM，够本函数使用） */
  const makeImageData = (n: number) =>
    ({ data: new Uint8ClampedArray(n * 4) }) as unknown as ImageData;

  it('segPixels 短于 pixels 时，超出部分保持灰度且不出现 NaN', () => {
    const pixels = [100, 100, 100, 100];
    // 模拟「标签体数据形状与 MRI 不一致」时抽出来的残缺切片
    const segPixels = [1, 1];
    const imageData = makeImageData(4);

    renderSlicePixels(pixels, segPixels, imageData, viewSettings);

    const data = imageData.data;
    // 前两个像素应叠加了 NCR 的蓝色 #64B5F6（opacity=100 时完全取代灰度）
    expect(data[0]).toBe(0x64);
    expect(data[1]).toBe(0xb5);
    expect(data[2]).toBe(0xf6);
    // 后两个像素应为纯灰度（R=G=B），且不是 NaN 转成的脏值
    expect(data[8]).toBe(data[9]);
    expect(data[9]).toBe(data[10]);
    for (let i = 0; i < data.length; i++) {
      expect(Number.isNaN(data[i])).toBe(false);
    }
  });
});

describe('getVolumeDims —— 维度兜底的唯一来源', () => {
  const real: NiftiData = {
    header: { dims: [3, 240, 240, 155] } as NiftiData['header'],
    image: new ArrayBuffer(0),
    typedArray: new Float32Array(0),
  };

  it('有合法维度时原样返回', () => {
    expect(getVolumeDims(real)).toEqual([3, 240, 240, 155]);
  });

  it('未加载（null）时回退到 DEFAULT_VOLUME_DIMS', () => {
    expect(getVolumeDims(null)).toEqual([0, 240, 240, 155]);
  });

  it('未加载（undefined）时回退到 DEFAULT_VOLUME_DIMS', () => {
    expect(getVolumeDims(undefined)).toEqual([0, 240, 240, 155]);
  });

  it('维度数组不足 4 项时回退（防止 header 异常导致按不存在形状读取）', () => {
    const bad: NiftiData = {
      header: { dims: [3, 240] } as NiftiData['header'],
      image: new ArrayBuffer(0),
      typedArray: new Float32Array(0),
    };
    expect(getVolumeDims(bad)).toEqual([0, 240, 240, 155]);
  });
});

describe('getMaxSlices —— 三平面最大切片索引（各轴减 1）', () => {
  it('标准 BraTS 尺寸：axial=154, coronal=239, sagittal=239', () => {
    expect(getMaxSlices([3, 240, 240, 155])).toEqual({
      axial: 154,
      coronal: 239,
      sagittal: 239,
    });
  });

  it('某轴缺失时按 DEFAULT_VOLUME_DIMS 兜底（axial 用 155-1）', () => {
    expect(getMaxSlices([3, 240, 240])).toEqual({
      axial: 154,
      coronal: 239,
      sagittal: 239,
    });
  });

  it('各平面键名与切片方向对齐（axial→Z，coronal→Y，sagittal→X）', () => {
    const m = getMaxSlices([3, 10, 20, 30]);
    expect(m.axial).toBe(29); // nz-1
    expect(m.coronal).toBe(19); // ny-1
    expect(m.sagittal).toBe(9); // nx-1
  });
});
