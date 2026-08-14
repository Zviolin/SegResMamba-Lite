"""
【模型模块】
提供统一的模型接口
参考 Baseline_Version/model.py 实现

支持的模型：
- UNet: 经典编码器-解码器结构
- SwinUNETR: 基于 Swin Transformer 的 U-Net
- SegResNet: NVIDIA BraTS 冠军模型
"""

from monai.networks.nets import UNet, SwinUNETR, SegResNet, SegResNetDS


def get_model(
    model_name="segresnet",
    in_channels=4,
    out_channels=4,
    device="cuda",
    use_deep_supervision=False,
):
    """
    获取模型实例

    Args:
        model_name: 模型名称 ("unet", "swin_unetr", "segresnet")
        in_channels: 输入通道数
        out_channels: 输出通道数
        device: 设备
        use_deep_supervision: 是否使用深层监督（仅 SegResNet 支持）
    """
    if model_name == "unet":
        model = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        ).to(device)
    elif model_name == "swin_unetr":
        model = SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=24,
            use_checkpoint=True,
        ).to(device)
    elif model_name == "segresnet":
        if use_deep_supervision:
            model = SegResNetDS(
                spatial_dims=3,
                in_channels=in_channels,
                out_channels=out_channels,
                init_filters=16,
                dsdepth=3,
            ).to(device)
        else:
            model = SegResNet(
                spatial_dims=3,
                in_channels=in_channels,
                out_channels=out_channels,
                init_filters=16,
                dropout_prob=0.2,
            ).to(device)
    else:
        raise ValueError(f"未知模型名称: {model_name}")

    return model


__all__ = [
    "get_model",
    "UNet",
    "SwinUNETR",
    "SegResNet",
    "SegResNetDS",
]