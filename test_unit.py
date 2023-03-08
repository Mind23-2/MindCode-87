
def TestTSN():
    from mindspore import context, Tensor
    import os
    device_id = int(os.getenv('DEVICE_ID', '0'))
    context.set_context(mode=context.PYNATIVE_MODE, enable_auto_mixed_precision=True,
                        device_target='Ascend', save_graphs=False, device_id=device_id)
    context.set_context(mode=context.PYNATIVE_MODE, enable_auto_mixed_precision=True,
                        device_target='CPU', save_graphs=False, device_id=device_id)
    from models import TSN
    model = TSN(3, 101)
    import numpy as np
    x = np.ones((1,3,224,224), dtype=np.float32)
    x=Tensor(x)
    y=model(x)
    print('输出y: ', y, y.shape)
    pass


TestTSN()

