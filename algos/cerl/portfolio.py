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

	if portfolio_id == 10:
		# Learner 1
		portfolio.append(
			Learner(model_constructor, args, gamma=0.9))

		# Learner 2
		portfolio.append(
			Learner(model_constructor, args, gamma=0.99))

		# Learner 3
		portfolio.append(
			Learner(model_constructor, args, gamma=0.999))


	if portfolio_id == 20:
		# Learner 1
		portfolio.append(
			Learner(model_constructor, args, 0.9, **sac_specific))

		# Learner 2
		portfolio.append(
			Learner(model_constructor, args, 0.99, **sac_specific))


		# Learner 3
		portfolio.append(
			Learner(model_constructor, args, 0.999, **sac_specific))



	return portfolio
