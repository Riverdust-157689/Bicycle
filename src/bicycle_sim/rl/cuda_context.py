"""One Warp-owned CUDA stream for graph capture, task tensors and learning."""

from contextlib import contextmanager


@contextmanager
def shared_cuda_stream(device):
    """Use a native Warp stream and expose it to Torch, in this direction.

    Wrapping Torch's default stream and installing it as Warp's current stream
    can fail at mjlab's graph capture with "unknown stream". Keep the native
    Warp stream alive for the entire simulation/learning scope instead.
    """
    import torch
    import warp as wp

    torch.cuda.set_device(device)
    wp.init()
    with wp.ScopedDevice(device):
        warp_stream = wp.Stream(device=device)
        torch_stream = wp.stream_to_torch(warp_stream)
        caller = torch.cuda.current_stream(device)
        torch_stream.wait_stream(caller)
        with wp.ScopedStream(warp_stream), torch.cuda.stream(torch_stream):
            try:
                yield
            finally:
                # Finish work before the Warp stream's owner leaves this scope.
                wp.synchronize_stream(warp_stream)
        caller.wait_stream(torch_stream)
