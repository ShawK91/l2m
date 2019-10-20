import numpy as np
from collections import OrderedDict
import math



def get_core_matrix(yaw):
    core_matrix = np.zeros(shape=(3, 3))
    core_matrix[0][0] = math.cos(yaw)
    core_matrix[0][2] = -1.0 * math.sin(yaw)
    core_matrix[1][1] = 1
    core_matrix[2][0] = math.sin(yaw)
    core_matrix[2][2] = math.cos(yaw)
    return core_matrix

def shape_observation(state_desc):
    o = OrderedDict()
    for body_part in [
            'pelvis', 'femur_r', 'tibia_r', 'toes_r', 'femur_l',
            'tibia_l', 'talus_l', 'calcn_l', 'toes_l', 'torso', 'head'
    ]:
        # position
        o[body_part + '_x'] = state_desc['body_pos'][body_part][0]
        o[body_part + '_y'] = state_desc['body_pos'][body_part][1]
        o[body_part + '_z'] = state_desc['body_pos'][body_part][2]
        # velocity
        o[body_part + '_v_x'] = state_desc["body_vel"][body_part][0]
        o[body_part + '_v_y'] = state_desc["body_vel"][body_part][1]
        o[body_part + '_v_z'] = state_desc["body_vel"][body_part][2]

        o[body_part + '_x_r'] = state_desc["body_pos_rot"][body_part][0]
        o[body_part + '_y_r'] = state_desc["body_pos_rot"][body_part][1]
        o[body_part + '_z_r'] = state_desc["body_pos_rot"][body_part][2]

        o[body_part + '_v_x_r'] = state_desc["body_vel_rot"][body_part][0]
        o[body_part + '_v_y_r'] = state_desc["body_vel_rot"][body_part][1]
        o[body_part + '_v_z_r'] = state_desc["body_vel_rot"][body_part][2]

    for joint in [
            'hip_r', 'knee_r', 'ankle_r', 'hip_l', 'knee_l', 'ankle_l',
            'back'
    ]:
        if 'hip' not in joint:
            o[joint + '_joint_pos'] = state_desc['joint_pos'][joint][0]
            o[joint + '_joint_vel'] = state_desc['joint_vel'][joint][0]
        else:
            for i in range(3):
                o[joint + '_joint_pos_' +
                  str(i)] = state_desc['joint_pos'][joint][i]
                o[joint + '_joint_vel_' +
                  str(i)] = state_desc['joint_vel'][joint][i]

    # In NIPS2017, only use activation
    for muscle in sorted(state_desc["muscles"].keys()):
        activation = state_desc["muscles"][muscle]["activation"]
        if isinstance(activation, float):
            activation = [activation]
        for i, val in enumerate(activation):
            o[muscle + '_activation_' + str(i)] = activation[i]

        fiber_length = state_desc["muscles"][muscle]["fiber_length"]
        if isinstance(fiber_length, float):
            fiber_length = [fiber_length]
        for i, val in enumerate(fiber_length):
            o[muscle + '_fiber_length_' + str(i)] = fiber_length[i]

        fiber_velocity = state_desc["muscles"][muscle]["fiber_velocity"]
        if isinstance(fiber_velocity, float):
            fiber_velocity = [fiber_velocity]
        for i, val in enumerate(fiber_velocity):
            o[muscle + '_fiber_velocity_' + str(i)] = fiber_velocity[i]

    # z axis of mass have some problem now, delete it later
    o['mass_x'] = state_desc["misc"]["mass_center_pos"][0]
    o['mass_y'] = state_desc["misc"]["mass_center_pos"][1]
    o['mass_z'] = state_desc["misc"]["mass_center_pos"][2]

    o['mass_v_x'] = state_desc["misc"]["mass_center_vel"][0]
    o['mass_v_y'] = state_desc["misc"]["mass_center_vel"][1]
    o['mass_v_z'] = state_desc["misc"]["mass_center_vel"][2]
    for key in ['talus_l_y', 'toes_l_y']:
        o['touch_indicator_' + key] = np.clip(0.05 - o[key] * 10 + 0.5, 0.,
                                              1.)
        o['touch_indicator_2_' + key] = np.clip(0.1 - o[key] * 10 + 0.5,
                                                0., 1.)

    # Tranformer
    core_matrix = get_core_matrix(o['pelvis_y_r'])
    pelvis_pos = np.array([o['pelvis_x'], o['pelvis_y'],
                           o['pelvis_z']]).reshape((3, 1))
    pelvis_vel = np.array(
        [o['pelvis_v_x'], o['pelvis_v_y'], o['pelvis_v_z']]).reshape((3,
                                                                      1))
    for body_part in [
            'pelvis', 'femur_r', 'tibia_r', 'toes_r', 'femur_l',
            'tibia_l', 'talus_l', 'calcn_l', 'toes_l', 'torso', 'head'
    ]:
        # rotation
        if body_part != 'mass':
            o[body_part + '_y_r'] -= o['pelvis_y_r']
            o[body_part + '_v_y_r'] -= o['pelvis_v_y_r']
        # position/velocity
        global_pos = []
        global_vel = []
        for each in ['_x', '_y', '_z']:
            global_pos.append(o[body_part + each])
            global_vel.append(o[body_part + '_v' + each])
        global_pos = np.array(global_pos).reshape((3, 1))
        global_vel = np.array(global_vel).reshape((3, 1))
        pelvis_rel_pos = core_matrix.dot(global_pos - pelvis_pos)
        w = o['pelvis_v_y_r']
        offset = np.array(
            [-w * pelvis_rel_pos[2], 0, w * pelvis_rel_pos[0]])
        pelvis_rel_vel = core_matrix.dot(global_vel - pelvis_vel) + offset
        for i, each in enumerate(['_x', '_y', '_z']):
            o[body_part + each] = pelvis_rel_pos[i][0]
            o[body_part + '_v' + each] = pelvis_rel_vel[i][0]

    for key in ['pelvis_x', 'pelvis_z', 'pelvis_y_r']:
        del o[key]

    current_v = np.array(state_desc['body_vel']['pelvis']).reshape((3, 1))
    pelvis_current_v = core_matrix.dot(current_v)
    o['pelvis_v_x'] = pelvis_current_v[0]
    o['pelvis_v_z'] = pelvis_current_v[2]

    vals = list(o.values())
    res = []
    for v in vals:
        if not isinstance(v, float):
            for k in v:
                res.append(k)
        else:
            res.append(v)


    feet_dis = ((o['tibia_l_x'] - o['tibia_r_x'])**2 +
                (o['tibia_l_z'] - o['tibia_r_z'])**2)**0.5
    res.append(feet_dis)


    return np.array(res)

