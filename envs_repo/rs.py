import numpy as np



d_reward = {}
d_reward['weight'] = {}
d_reward['weight']['footstep'] = 10
d_reward['weight']['effort'] = 1
d_reward['weight']['v_tgt'] = 1

d_reward['alive'] = 0.1
d_reward['effort'] = 0

d_reward['footstep'] = {}
d_reward['footstep']['effort'] = 0
d_reward['footstep']['del_t'] = 0
d_reward['footstep']['del_v'] = 0


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
    reward -= np.linalg.norm(velocity_deviation)

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








#R!

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