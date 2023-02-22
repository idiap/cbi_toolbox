# Copyright (c) 2022 Idiap Research Institute, http://www.idiap.ch/
# Written by François Marelli <francois.marelli@idiap.ch>
#
# This file is part of CBI Toolbox.
#
# CBI Toolbox is free software: you can redistribute it and/or modify
# it under the terms of the 3-Clause BSD License.
#
# CBI Toolbox is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# 3-Clause BSD License for more details.
#
# You should have received a copy of the 3-Clause BSD License along
# with CBI Toolbox. If not, see https://opensource.org/licenses/BSD-3-Clause.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import pathlib
import json
import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("dir", type=str)

    args = parser.parse_args()

    root_path = pathlib.Path(args.dir)

    assert root_path.is_dir()

    for var_dir in root_path.iterdir():
        if not var_dir.is_dir():
            continue

        var_dict = {"variable": var_dir.name, "values": []}

        def dir_to_val(dirname):
            try:
                return float(dirname.name)
            except:
                return -1

        for val_dir in sorted(var_dir.iterdir(), key=dir_to_val):
            if not val_dir.is_dir():
                continue

            try:
                var_val = float(val_dir.name)
            except:
                var_val = val_dir.name

            results = {}

            for result in val_dir.glob("**/results.json"):
                with result.open("r") as fp:
                    res_dict = json.load(fp)

                for key, val in res_dict.items():
                    if key not in results:
                        results[key] = []
                    results[key].append(val)

            var_dict["values"].append(var_val)

            for key, val in results.items():
                mean = np.mean(val)
                std = np.std(val)

                if key not in var_dict:
                    var_dict[key] = []

                var_dict[key].append((mean, std))

        out_file = var_dir / "results"
        np.savez(str(out_file), **var_dict)
