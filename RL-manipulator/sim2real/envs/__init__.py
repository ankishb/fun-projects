# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from gym.envs.registration import register

register(
   id='ur_Env-v0',
   entry_point='envs.ur_Env:UrGymEnv',
)
