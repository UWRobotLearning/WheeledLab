# source/wheeledlab_tasks/drifting/disable_lidar.py
def disable_all_lidars(env, env_ids=None, **kwargs):
    """
    Hard-disable every RTX-LiDAR in the stage.
    Safe even if the sensor extension hasn't been loaded yet.
    """
    try:
        from omni.isaac.sensor import _sensor         
    except ModuleNotFoundError:
        return

    iface = _sensor.acquire_lidar_sensor_interface()
    for prim in iface.get_lidar_sensor_prims():
        iface.set_enabled(prim, False)
        iface.set_debug_vis(prim, False)
