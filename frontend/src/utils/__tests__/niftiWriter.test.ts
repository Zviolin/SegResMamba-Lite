/**
 * NIfTI 导出工具单测：验证 header 结构与 gzip 往返
 */
import { describe, it, expect } from 'vitest';
import { buildNiftiHeader, gzipBytes } from '../niftiWriter';

describe('buildNiftiHeader', () => {
  it('生成 348 字节标准 NIfTI-1 头，字段正确', () => {
    const header = buildNiftiHeader([240, 240, 155], [1.0, 1.0, 1.5]);
    expect(header.byteLength).toBe(348);
    const view = new DataView(header);
    expect(view.getInt32(0, true)).toBe(348);           // sizeof_hdr
    expect(view.getInt16(40, true)).toBe(3);            // dim[0]=3 维
    expect(view.getInt16(42, true)).toBe(240);          // dim[1]=nx
    expect(view.getInt16(44, true)).toBe(240);          // dim[2]=ny
    expect(view.getInt16(46, true)).toBe(155);          // dim[3]=nz
    expect(view.getInt16(70, true)).toBe(2);            // datatype=uint8
    expect(view.getInt16(72, true)).toBe(8);            // bitpix
    expect(view.getFloat32(80, true)).toBeCloseTo(1.0); // pixdim[1]
    expect(view.getFloat32(88, true)).toBeCloseTo(1.5); // pixdim[3]
    expect(view.getFloat32(108, true)).toBe(352);       // vox_offset
    // magic "n+1\0"
    expect(String.fromCharCode(view.getUint8(344), view.getUint8(345), view.getUint8(346), view.getUint8(347))).toBe('n+1\0');
  });

  it('支持任意尺寸的掩码', () => {
    const header = buildNiftiHeader([64, 64, 64], [2, 2, 2]);
    const view = new DataView(header);
    expect(view.getInt16(42, true)).toBe(64);
    expect(view.getInt16(46, true)).toBe(64);
  });
});

describe('gzipBytes', () => {
  it('压缩后能解压还原原始数据', async () => {
    const raw = new Uint8Array(1000);
    for (let i = 0; i < 1000; i++) raw[i] = i % 256;
    const gz = await gzipBytes(raw);
    // 解压验证（DecompressionStream 同为内置 API）
    const stream = new Blob([gz]).stream().pipeThrough(new DecompressionStream('gzip'));
    const buf = await new Response(stream).arrayBuffer();
    const back = new Uint8Array(buf);
    expect(back.length).toBe(1000);
    expect(back[0]).toBe(0);
    expect(back[999]).toBe(999 % 256);
  });
});
