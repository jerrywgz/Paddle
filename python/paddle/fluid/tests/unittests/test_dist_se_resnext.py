#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import unittest
from test_dist_base import TestDistBase


class TestDistSeResneXt2x2(TestDistBase):
    def test_se_resnext(self):
<<<<<<< HEAD
        # TODO(paddle-dev): Is the delta too large?
        self.check_with_place("dist_se_resnext.py", delta=0.2)
=======
        self.check_with_place("dist_se_resnext.py")
>>>>>>> 772ceee395088540be71ae97e0c3466846410c1d


if __name__ == "__main__":
    unittest.main()
