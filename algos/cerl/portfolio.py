# ******************************************************************************
# Copyright 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

from algos.cerl.learner import Learner


def initialize_portfolio(portfolio, args, genealogy, portfolio_id):
	"""Portfolio of learners

        Parameters:
            portfolio (list): Incoming list
            args (object): param class

        Returns:
            portfolio (list): Portfolio of learners
    """

	if portfolio_id == 10:

		# Learner 1
		wwid = genealogy.new_id('learner_1')
		portfolio.append(
			Learner(wwid, args.algo,args.state_dim, args.goal_dim, args.action_dim, actor_lr=args.actor_lr, critic_lr=args.critic_lr, gamma=0.9, tau=args.tau))

		# Learner 2
		wwid = genealogy.new_id('learner_3')
		portfolio.append(
			Learner(wwid, args.algo,args.state_dim, args.goal_dim, args.action_dim, actor_lr=args.actor_lr, critic_lr=args.critic_lr, gamma=0.99, tau=args.tau))

		# Learner 3
		wwid = genealogy.new_id('learner_4')
		portfolio.append(
			Learner(wwid, args.algo,args.state_dim, args.goal_dim, args.action_dim, actor_lr=args.actor_lr, critic_lr=args.critic_lr, gamma=0.997, tau=args.tau))

		# Learner 4
		wwid = genealogy.new_id('learner_4')
		portfolio.append(
			Learner(wwid, args.algo,args.state_dim, args.goal_dim, args.action_dim, actor_lr=args.actor_lr, critic_lr=args.critic_lr, gamma=0.9995, tau=args.tau))

		# Learner 5
		wwid = genealogy.new_id('learner_5')
		portfolio.append(
			Learner(wwid, args.algo, args.state_dim, args.goal_dim, args.action_dim, actor_lr=args.actor_lr, critic_lr=args.critic_lr, gamma=0.9999, tau=args.tau))







	return portfolio
