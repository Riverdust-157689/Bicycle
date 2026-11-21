"""Host allocation compatibility adapter for the locked MuJoCo-Warp backend.

The backend exposes no EPA horizon option in SimulationCfg. Its convex and
flex collision allocators import a Python constant, while kernels use the
allocated array's runtime shape. Configure before constructing Simulation:
CUDA graph replay retains the buffers allocated during capture. This changes
workspace capacity, not geometry, contact parameters or overflow handling.
The setting is process-wide; do not change it concurrently with graph capture.
"""


def configure_epa_horizon(capacity):
    from mujoco_warp._src import collision_convex, collision_flex

    if isinstance(capacity, bool) or not isinstance(capacity, int) or not 24 <= capacity <= 256:
        raise ValueError("epa_horizon_capacity must be an integer in [24, 256]")
    for module in (collision_convex, collision_flex):
        if not hasattr(module, "MJ_MAX_EPAHORIZON"):
            raise RuntimeError(
                "MuJoCo-Warp EPA allocator changed; review the compatibility adapter"
            )
    for module in (collision_convex, collision_flex):
        module.MJ_MAX_EPAHORIZON = capacity
    return capacity
