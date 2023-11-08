import torch

def pth2onnx():
    device = torch.device('cpu')
    print('using ' + str(device))
    from network.network_pmrid import PMRID
    model = PMRID()
    model.load_state_dict(torch.load('./checkpoints/SCUNet_100000_G.pth', map_location=torch.device('cpu')))
    model.eval()

    x = torch.rand(1, 1, 540, 960)  # 生成张量

    export_onnx_file = "./checkpoints/net_timm448.onnx"  # 目的ONNX文件名
    print('======> ', export_onnx_file)
    torch.onnx.export(model, x, export_onnx_file, opset_version=13, # opset_version=10
                      verbose=True,
                      do_constant_folding=True,  # 是否执行常量折叠优化
                      # do_constant_folding=False,  # 是否执行常量折叠优化
                      input_names=["input", "info"],    # 输入名
                      output_names=["output"],  # 输出名
                      )


# def export_openvino(model, file, half, prefix=colorstr('OpenVINO:')):
#     try:
#         check_requirements(('openvino-dev',))  # requires openvino-dev: https://pypi.org/project/openvino-dev/
#         import openvino.inference_engine as ie
#
#         LOGGER.info(f'\n{prefix} starting export with openvino {ie.__version__}...')
#         f = str(file).replace('.pt', f'_openvino_model{os.sep}')
#
#         cmd = f"mo --input_model {file.with_suffix('.onnx')} --output_dir {f} --data_type {'FP16' if half else 'FP32'}"
#         subprocess.check_output(cmd.split())  # export
#         with open(Path(f) / file.with_suffix('.yaml').name, 'w') as g:
#             yaml.dump({'stride': int(max(model.stride)), 'names': model.names}, g)  # add metadata.yaml
#
#         LOGGER.info(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
#         return f
#     except Exception as e:
#         LOGGER.info(f'\n{prefix} export failure: {e}')


if __name__ == '__main__':
    # save_model()
    pth2onnx()

