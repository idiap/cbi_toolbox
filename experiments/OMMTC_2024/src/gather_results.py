# Copyright (c) 2023 Idiap Research Institute, http://www.idiap.ch/
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

        print(f"Gathering {str(var_dir)}")
        var_dict = {"variable": var_dir.name, "values": []}
        empty = True

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

            for result in val_dir.glob("**/results.npz"):
                res_dict = np.load(result)

                for key in res_dict.files:
                    if key not in results:
                        results[key] = []
                    results[key].append(res_dict[key])

            if len(results) == 0:
                continue

            empty = False

            var_dict["values"].append(var_val)
            var_dict["N"] = []

            for key, val in results.items():
                var_dict["N"].append(len(val))

                if key not in var_dict:
                    var_dict[key] = []

                if key == "solution":
                    var_dict[key].append(val)

                else:
                    mean = np.mean(val, 0)
                    std = np.std(val, 0)

                    var_dict[key].append((mean, std))

        if not empty:
            out_file = var_dir / "combined"
            np.savez(str(out_file), **var_dict)
        else:
            print("Warning: no results found")
