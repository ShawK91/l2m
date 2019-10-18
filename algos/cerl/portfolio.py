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


def initialize_portfolio(portfolio, args, genealogy, portfolio_id, model_constructor):
	"""Portfolio of learners

        Parameters:
            portfolio (list): Incoming list
            args (object): param class

        Returns:
            portfolio (list): Portfolio of learners
    """

	sac_specific = {}
	sac_specific['autotune'] = args.autotune
	sac_specific['entropy'] = True

	if portfolio_id == 10:
		# Learner 1
		portfolio.append(
			Learner(model_constructor, args, args.algo, gamma=0.92, **sac_specific))

		# Learner 2
		portfolio.append(
			Learner(model_constructor, args, args.algo, gamma=0.97, **sac_specific))

		# Learner 3
		portfolio.append(
			Learner(model_constructor, args, args.algo, gamma=0.994, **sac_specific))




	if portfolio_id == 20:
		# Learner 1
		portfolio.append(
			Learner(model_constructor, args, 'sac', 0.92, **sac_specific))
		portfolio[-1].algo.actor.stochastic = True

		# Learner 2
		portfolio.append(
			Learner(model_constructor, args, 'sac', 0.97, **sac_specific))
		portfolio[-1].algo.actor.stochastic = True

		sac_specific['entropy'] = False
		sac_specific['autotune'] = False
		# Learner 3
		portfolio.append(
			Learner(model_constructor, args, 'sac', 0.97, **sac_specific))
		portfolio[-1].algo.actor.stochastic = False

		# Learner 4
		portfolio.append(
			Learner(model_constructor, args, 'sac', 0.994, **sac_specific))
		portfolio[-1].algo.actor.stochastic = False


	return portfolio
