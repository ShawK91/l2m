


import numpy as np

def crouch(pelvis_y):
    """pelvis remains below 0.8m (crouched position)
        Parameters:
            pelvis_y (ndarray): pelvis positions in y
        Returns:
            r (float): reward within the range (0.0,1.0) based on percent of timeseteps that the constraint was satisfied
    """

    r = pelvis_y < 0.85
    return r

def knee_bend(ltibia_angle, lfemur_angle, rtibia_angle, rfemur_angle):
    """knee remains bend
        Parameters:
            ltibia_angle (ndarray): angle for left tibia in degrees
            lfemur_angle (ndarray): angle for left femur in degrees
            rtibia_angle (ndarray): angle for right tibia in degrees
            rfemur_angle (ndarray): angle for right femur in degrees
        Returns:
            r (float): reward within the range (0.0,1.0) based on percent of timeseteps that the constraint was satisfied
    """

    tibia_bent_back = ltibia_angle<0 and rtibia_angle<0
    knee_bend = ltibia_angle < lfemur_angle and rtibia_angle < rfemur_angle
    r = tibia_bent_back and knee_bend
    return r

def vel_follower(self): # for L2M2019 Round 2
    state_desc = self.get_state_desc()

    # reward from velocity (penalize from deviating from v_tgt)
    p_body = [state_desc['body_pos']['pelvis'][0], -state_desc['body_pos']['pelvis'][2]]
    v_body = [state_desc['body_vel']['pelvis'][0], -state_desc['body_vel']['pelvis'][2]]
    v_tgt = self.vtgt.get_vtgt(p_body).T

    x_penalty = -(v_tgt[0,0]-v_body[0])**2
    z_penalty = -(v_tgt[0,1]-v_body[1])**2

    #r = -np.linalg.norm(v_body - v_tgt)
    return x_penalty, z_penalty


def toes_low(ltoe, rtoe):
    """foot is not raised too high
        Parameters:
            lfoot (ndarray): left foot positions in y
            rfoot (ndarray): right foot positions in y
        Returns:
            r (float): continous reward based on degree that the constraint was satisfied
    """

    r = ltoe < 0.3 and rtoe < 0.3
    return float(r)


################# ROUND 2 BRS #######################


def pelvis_swing(pel_v, use_synthetic_targets, phase_len):


    if use_synthetic_targets:
        mask = np.ones(phase_len*4+1)
        mask[0:30] = 0
        mask[phase_len-5:phase_len+30] = 0
        mask[2*phase_len-5:2*phase_len+30] = 0
        mask[3*phase_len-5:3*phase_len+30] = 0


    else:
        mask = np.ones(1001)
        mask[0:40] = 0
        mask[295:340] = 0
        mask[595:640] = 0
        mask[895:940] = 0

    """Penalizes pelvis swing (a dynamic shaping function)
        Parameters:
            pel_v (ndarray): pelvis velocity trajectory
        Returns:
            r (float): continous reward based on degree that the constraint was satisfied
    """

    pelvis_swing = np.square(np.ediff1d(pel_v[:,0]))
    mask = mask[0:len(pelvis_swing)]
    r = -np.sum(np.multiply(mask, pelvis_swing)) * 50


    return r


def knee_bend_trajectory(ltibia_angle, lfemur_angle, rtibia_angle, rfemur_angle):
    """knee remains bend
        Parameters:
            ltibia_angle (ndarray): angle for left tibia in degrees
            lfemur_angle (ndarray): angle for left femur in degrees
            rtibia_angle (ndarray): angle for right tibia in degrees
            rfemur_angle (ndarray): angle for right femur in degrees
        Returns:
            r (float): reward within the range (0.0,1.0) based on percent of timeseteps that the constraint was satisfied
    """

    tibia_bent_back = np.bitwise_and((ltibia_angle<0), (rtibia_angle<0))
    knee_bend = np.bitwise_and((ltibia_angle < lfemur_angle), (rtibia_angle < rfemur_angle))
    r = np.bitwise_and(tibia_bent_back, knee_bend)
    r = np.mean(r)
    return r


def foot_z_rs(lfoot, rfoot):
    """Foot do not criss-cross over each other in z-axis
        Parameters:
            lfoot (ndarray): left foot positions in z
            rfoot (ndarray): right foot positions in z
        Returns:
            r (float): reward within the range (0.0,1.0) based on percent of timeseteps that the constraint was satisfied
    """
    r = np.mean(lfoot[:,2] < rfoot[:,2])
    return r




def knee_bend_regression(ltibia_angle, lfemur_angle, rtibia_angle, rfemur_angle):
    """knee remains bend (soft-constraint)
        Parameters:
            ltibia_angle (ndarray): angle for left tibia in degrees
            lfemur_angle (ndarray): angle for left femur in degrees
            rtibia_angle (ndarray): angle for right tibia in degrees
            rfemur_angle (ndarray): angle for right femur in degrees
        Returns:
            r (float): continous reward based on the degree that the constraint was satisfied
    """

    tibia_bent_back = -np.sum(ltibia_angle) - np.sum(rtibia_angle)
    knee_bend = np.sum(lfemur_angle-ltibia_angle) + np.sum(rfemur_angle-rtibia_angle)
    r  = tibia_bent_back + knee_bend
    return r


def thighs_swing(lfemur_angle, rfemur_angle):
    """rewards thighs swing (a dynamic shaping function)
        Parameters:
            lfemur_angle (ndarray): angle for left femur in degrees
            rfemur_angle (ndarray): angle for right femur in degrees
        Returns:
            r (float): continous reward based on degree that the constraint was satisfied
    """

    lswing = np.sum(np.abs(np.ediff1d(lfemur_angle)))
    rswing = np.sum(np.abs(np.ediff1d(rfemur_angle)))

    r = lswing + rswing
    return r

def head_behind_pelvis(head_x):
    """head remains behind pelvis
        Parameters:
            head_x (ndarray): head position in x relative to pelvis x
        Returns:
            r (float): reward within the range (0.0,1.0) based on percent of timeseteps that the constraint was satisfied
    """

    r = np.mean(head_x < 0)
    return r




def pelvis_slack(pelvis_y):
    """slack continous measurement for pelvis remains below 0.8m (crouched position)
        Parameters:
            pelvis_y (ndarray): pelvis positions in y
        Returns:
            r (float): continous reward based on degree that the constraint was satisfied
    """
    r = len(pelvis_y) - np.sum(np.abs(pelvis_y-0.75))
    return r

def foot_y(lfoot, rfoot):
    """foot is not raised too high
        Parameters:
            lfoot (ndarray): left foot positions in y
            rfoot (ndarray): right foot positions in y
        Returns:
            r (float): continous reward based on degree that the constraint was satisfied
    """

    r = lfoot[:,1] + rfoot[:,1]
    r = len(r) - 2.0 *np.sum(r)
    return r



def final_footx(pelv_x, lfoot, rfoot):
    """slack continous measurement for final foot position without raising it
        Parameters:
            pelv_x (ndarray): pelvis positions in x
            lfoot (ndarray): left foot positions in y
            rfoot (ndarray): right foot positions in y
        Returns:
            r (float): continous reward based on degree that the constraint was satisfied
    """

    best_lfoot = (pelv_x + lfoot[:,0]) * (lfoot[:,1] < 0.1)
    best_rfoot = (pelv_x + rfoot[:,0]) * (rfoot[:,1] < 0.1)

    r = max(np.max(best_lfoot), np.max(best_rfoot))
    return r


def shaped_data(s, r, footz_w, kneefoot_w, pelv_w, footy_w, head_w):
    """method to shape a reward based on the above constraints (behavioral reward shaping unlike the temporal one)
        Parameters:
                s (ndarray): Current State
                r (ndarray): Reward
                footz_w (float): weight for computing shaped reward
                kneefoot_w (float): weight for computing shaped reward
                pelv_w (float): weight for computing shaped reward
                footy_w (float): weight for computing shaped reward
                head_w (float): weight for computing shaped reward
        Returns:
            r (ndarray): shaped reward with behavioral shaping
    """
    ####### FOOT Z AXIS ######
    footz_flag = np.where(s[:,95] > s[:,83]) #Left foot z greater than right foot z

    ####### KNEE BEFORE FOOT #######
    kneebend_flag = np.where(np.bitwise_or((s[:,125] > s[:,104]), (s[:,119]>s[:,107])))
    tibia_bent_back_flag = np.where(np.bitwise_or( (s[:,125]>0),  (s[:,119]>0)))

    ######## PELVIS BELOW 0.8 #######
    pelv_flag = np.where(s[:,79] > 0.8)

    ######### FOOT HEIGHT ######
    footy_flag = np.where(np.bitwise_or(s[:,94]>0.15, s[:,82]>0.15))

    ######## HEAD BEHIND PELVIS #######
    head_flag = np.where((s[:,75]>0))

    ##### INCUR PENALTIES #####
    r[footz_flag] = r[footz_flag] + footz_w

    r[kneebend_flag] = r[kneebend_flag] + kneefoot_w
    r[tibia_bent_back_flag] = r[tibia_bent_back_flag] + kneefoot_w

    r[pelv_flag] = r[pelv_flag] + pelv_w
    r[footy_flag] = r[footy_flag] + footy_w
    r[head_flag] = r[head_flag] + head_w

    return r


def r2_shaped_data(s, r):
    """method to shape a reward based on the above constraints (behavioral reward shaping unlike the temporal one)
        Parameters:
                s (ndarray): Current State
                r (ndarray): Reward
                footz_w (float): weight for computing shaped reward
                kneefoot_w (float): weight for computing shaped reward
                pelv_w (float): weight for computing shaped reward
                footy_w (float): weight for computing shaped reward
                head_w (float): weight for computing shaped reward
        Returns:
            r (ndarray): shaped reward with behavioral shaping
    """
    ####### FOOT Z AXIS ######
    footz_flag = np.where(s[:,95] > s[:,83]) #Left foot z greater than right foot z


    ##### INCUR PENALTIES #####
    r[footz_flag] = r[footz_flag] - 5.0

    return r









################### INDICES ############

#["body_pos"]["tibia_l"][0] = 90
#["body_pos"]["pros_tibia_r"][0] = 84


#["body_pos"]["toes_l"][0] = 93
#["body_pos"]["pros_foot_r"][0] = 81

#["body_pos"]["toes_l"][1] = 94
#["body_pos"]["toes_l"][2] = 95

#["body_pos"]["pros_tibia_r"][1] = 85
#["body_pos"]["pros_tibia_r"][2] = 86

#obs_dict["body_pos"]["pelvis"][0] = 78
#obs_dict["body_pos"]["pelvis"][1] = 79


#['body_pos_rot']['tibia_l'][2] = 125
#['body_pos_rot']['pros_tibia_r'][2] = 119
#['body_pos_rot']['femur_l'][2] = 104
#['body_pos_rot']['femur_r'][2] = 107

#['body_pos']['head'][0] - 75

#['body_vel']['pelvis'][0] = 144
#['body_vel']['pelvis'][2] = 146









d_reward = {}
d_reward['weight'] = {}
d_reward['weight']['footstep'] = 5
d_reward['weight']['effort'] = 1
d_reward['weight']['v_tgt'] = 1

d_reward['alive'] = 0.1
d_reward['effort'] = 0

d_reward['footstep'] = {}
d_reward['footstep']['effort'] = 1
d_reward['footstep']['del_t'] = 0
d_reward['footstep']['del_v'] = 0
VEL_MULTIPLIER = 2.0




def get_reward_2(self): # for L2M2019 Round 2
    state_desc = self.get_state_desc()
    if not self.get_prev_state_desc():
        return 0

    reward = 0
    dt = self.osim_model.stepsize

    # alive reward
    # should be large enough to search for 'success' solutions (alive to the end) first
    reward += self.d_reward['alive']

    # effort ~ muscle fatigue ~ (muscle activation)^2
    ACT2 = 0
    for muscle in sorted(state_desc['muscles'].keys()):
        ACT2 += np.square(state_desc['muscles'][muscle]['activation'])
    self.d_reward['effort'] += ACT2*dt
    self.d_reward['footstep']['effort'] += ACT2*dt
    self.d_reward['footstep']['del_t'] += dt

    # reward from velocity (penalize from deviating from v_tgt)
    p_body = [state_desc['body_pos']['pelvis'][0], -state_desc['body_pos']['pelvis'][2]]
    v_body = [state_desc['body_vel']['pelvis'][0], -state_desc['body_vel']['pelvis'][2]]
    v_tgt = self.vtgt.get_vtgt(p_body).T

    self.d_reward['footstep']['del_v'] += (v_body - v_tgt)*dt

    # simulation ends successfully
    flag_success = (not self.is_done() # model did not fall down
        and (self.osim_model.istep >= self.spec.timestep_limit) # reached end of simulatoin
        and self.footstep['n'] > 5) # took more than 5 footsteps (to prevent standing still)

    # footstep reward (when made a new step)
    if self.footstep['new'] or flag_success:
        # footstep reward: so that solution does not avoid making footsteps
        # scaled by del_t, so that solution does not get higher rewards by making unnecessary (small) steps
        reward_footstep_0 = self.d_reward['weight']['footstep']*self.d_reward['footstep']['del_t']

        # deviation from target velocity
        # the average velocity a step (instead of instantaneous velocity) is used
        # as velocity fluctuates within a step in normal human walking
        #reward_footstep_v = -self.reward_w['v_tgt']*(self.footstep['del_vx']**2)
        reward_footstep_v = -self.d_reward['weight']['v_tgt']*np.linalg.norm(self.d_reward['footstep']['del_v'])/self.LENGTH0

        # panalize effort
        reward_footstep_e = -self.d_reward['weight']['effort']*self.d_reward['footstep']['effort']

        self.d_reward['footstep']['del_t'] = 0
        self.d_reward['footstep']['del_v'] = 0
        self.d_reward['footstep']['effort'] = 0

        reward += reward_footstep_0 + reward_footstep_v + reward_footstep_e

    # task bonus: if stayed enough at the first target
    if self.flag_new_v_tgt_field:
        reward += 500

    return reward


def get_reward_footsteps_r2(self):
    state_desc = self.get_state_desc()
    if not self.get_prev_state_desc():
        return 0

    reward = 0
    dt = self.osim_model.stepsize

    # alive reward
    # should be large enough to search for 'success' solutions (alive to the end) first
    reward += d_reward['alive']

    # effort ~ muscle fatigue ~ (muscle activation)^2
    ACT2 = 0
    for muscle in sorted(state_desc['muscles'].keys()):
        ACT2 += np.square(state_desc['muscles'][muscle]['activation'])
    d_reward['effort'] += ACT2 * dt
    d_reward['footstep']['effort'] += ACT2 * dt
    d_reward['footstep']['del_t'] += dt

    # reward from velocity (penalize from deviating from v_tgt)
    p_body = [state_desc['body_pos']['pelvis'][0], -state_desc['body_pos']['pelvis'][2]]
    v_body = [state_desc['body_vel']['pelvis'][0], -state_desc['body_vel']['pelvis'][2]]
    v_tgt = self.vtgt.get_vtgt(p_body).T

    velocity_deviation = (v_body - v_tgt) * dt
    reward -= np.linalg.norm(velocity_deviation)*VEL_MULTIPLIER

    # simulation ends successfully
    flag_success = (not self.is_done() # model did not fall down
        and (self.osim_model.istep >= self.spec.timestep_limit) # reached end of simulatoin
        and self.footstep['n'] > 5) # took more than 5 footsteps (to prevent standing still)


    # footstep reward (when made a new step)
    if self.footstep['new'] or flag_success:
        # footstep reward: so that solution does not avoid making footsteps
        # scaled by del_t, so that solution does not get higher rewards by making unnecessary (small) steps
        reward_footstep_0 = d_reward['weight']['footstep']# * d_reward['footstep']['del_t']

        # deviation from target velocity
        # the average velocity a step (instead of instantaneous velocity) is used
        # as velocity fluctuates within a step in normal human walking
        # reward_footstep_v = -self.reward_w['v_tgt']*(self.footstep['del_vx']**2)
        # reward_footstep_v = -d_reward['weight']['v_tgt'] * np.linalg.norm(
        #     velocity_deviation) / self.LENGTH0

        # panalize effort
        reward_footstep_e = -d_reward['weight']['effort'] * d_reward['footstep']['effort']

        d_reward['footstep']['del_t'] = 0
        d_reward['footstep']['effort'] = 0

        reward += reward_footstep_0 + reward_footstep_e

    # task bonus: if stayed enough at the first target
    if self.flag_new_v_tgt_field:
        reward += 500


    return reward






def get_reward_footsteps_r1(self):
    state_desc = self.get_state_desc()
    if not self.get_prev_state_desc():
        return 0

    reward = 0
    dt = self.osim_model.stepsize

    # alive reward
    # should be large enough to search for 'success' solutions (alive to the end) first
    reward += d_reward['alive']

    # effort ~ muscle fatigue ~ (muscle activation)^2
    ACT2 = 0
    for muscle in sorted(state_desc['muscles'].keys()):
        ACT2 += np.square(state_desc['muscles'][muscle]['activation'])
    d_reward['effort'] += ACT2 * dt
    d_reward['footstep']['effort'] += ACT2 * dt

    d_reward['footstep']['del_t'] += dt

    # reward from velocity (penalize from deviating from v_tgt)
    p_body = [state_desc['body_pos']['pelvis'][0], -state_desc['body_pos']['pelvis'][2]]
    v_body = [state_desc['body_vel']['pelvis'][0], -state_desc['body_vel']['pelvis'][2]]
    v_tgt = self.vtgt.get_vtgt(p_body).T

    velocity_deviation = (v_body - v_tgt) * dt

    reward -= np.linalg.norm(velocity_deviation)



    # footstep reward (when made a new step)
    if self.footstep['new']:
        # footstep reward: so that solution does not avoid making footsteps
        # scaled by del_t, so that solution does not get higher rewards by making unnecessary (small) steps
        reward_footstep_0 = d_reward['weight']['footstep']# * d_reward['footstep']['del_t']

        # deviation from target velocity
        # the average velocity a step (instead of instantaneous velocity) is used
        # as velocity fluctuates within a step in normal human walking
        # reward_footstep_v = -self.reward_w['v_tgt']*(self.footstep['del_vx']**2)
        # reward_footstep_v = -d_reward['weight']['v_tgt'] * np.linalg.norm(
        #     velocity_deviation) / self.LENGTH0

        # panalize effort
        reward_footstep_e = -d_reward['weight']['effort'] * d_reward['footstep']['effort']

        d_reward['footstep']['del_t'] = 0
        d_reward['footstep']['effort'] = 0

        reward += reward_footstep_0 + reward_footstep_e

    # # success bonus
    # if not self.is_done() and (
    #         self.osim_model.istep >= self.spec.timestep_limit):  # and self.failure_mode is 'success':
    #     # retrieve reward (i.e. do not penalize for the simulation terminating in a middle of a step)
    #     # reward_footstep_0 = self.d_reward['weight']['footstep']*self.d_reward['footstep']['del_t']
    #     # reward += reward_footstep_0 + 100
    #     reward += 10

    return reward


def get_reward_1(self):
    state_desc = self.get_state_desc()
    if not self.get_prev_state_desc():
        return 0

    reward = 0
    dt = self.osim_model.stepsize

    # alive reward
    # should be large enough to search for 'success' solutions (alive to the end) first
    reward += self.d_reward['alive']

    # effort ~ muscle fatigue ~ (muscle activation)^2
    ACT2 = 0
    for muscle in sorted(state_desc['muscles'].keys()):
        ACT2 += np.square(state_desc['muscles'][muscle]['activation'])
    self.d_reward['effort'] += ACT2 * dt
    self.d_reward['footstep']['effort'] += ACT2 * dt

    self.d_reward['footstep']['del_t'] += dt

    # reward from velocity (penalize from deviating from v_tgt)

    p_body = [state_desc['body_pos']['pelvis'][0], -state_desc['body_pos']['pelvis'][2]]
    v_body = [state_desc['body_vel']['pelvis'][0], -state_desc['body_vel']['pelvis'][2]]
    v_tgt = self.vtgt.get_vtgt(p_body).T

    self.d_reward['footstep']['del_v'] += (v_body - v_tgt) * dt

    # footstep reward (when made a new step)
    if self.footstep['new']:
        # footstep reward: so that solution does not avoid making footsteps
        # scaled by del_t, so that solution does not get higher rewards by making unnecessary (small) steps
        reward_footstep_0 = self.d_reward['weight']['footstep'] * self.d_reward['footstep']['del_t']

        # deviation from target velocity
        # the average velocity a step (instead of instantaneous velocity) is used
        # as velocity fluctuates within a step in normal human walking
        # reward_footstep_v = -self.reward_w['v_tgt']*(self.footstep['del_vx']**2)
        reward_footstep_v = -self.d_reward['weight']['v_tgt'] * np.linalg.norm(
            self.d_reward['footstep']['del_v']) / self.LENGTH0

        # panalize effort
        reward_footstep_e = -self.d_reward['weight']['effort'] * self.d_reward['footstep']['effort']

        self.d_reward['footstep']['del_t'] = 0
        self.d_reward['footstep']['del_v'] = 0
        self.d_reward['footstep']['effort'] = 0

        reward += reward_footstep_0 + reward_footstep_v + reward_footstep_e

    # success bonus
    if not self.is_done() and (
            self.osim_model.istep >= self.spec.timestep_limit):  # and self.failure_mode is 'success':
        # retrieve reward (i.e. do not penalize for the simulation terminating in a middle of a step)
        # reward_footstep_0 = self.d_reward['weight']['footstep']*self.d_reward['footstep']['del_t']
        # reward += reward_footstep_0 + 100
        reward += 10

    return reward