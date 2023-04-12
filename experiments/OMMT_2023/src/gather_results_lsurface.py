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
import json
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("dir", type=str)
    parser.add_argument("--experiment", action="store_true")

    args = parser.parse_args()

    root_path = pathlib.Path(args.dir)

    assert root_path.is_dir()

    def dir_to_val(dirname):
        try:
            return float(dirname.name)
        except:
            return -1

    l1_dict = {
        "values": [],
        "results": [],
    }

    if args.experiment:
        glob = "**/stats.json"
    else:
        glob = "**/results.json"

    for l1_dir in sorted(root_path.iterdir(), key=dir_to_val):
        if not l1_dir.is_dir():
            continue

        l1_val = float(l1_dir.name)

        for var_dir in l1_dir.iterdir():
            if not var_dir.is_dir():
                continue

            var_dict = {"variable": var_dir.name, "values": []}
            empty = True

            for val_dir in sorted(var_dir.iterdir(), key=dir_to_val):
                if not val_dir.is_dir():
                    continue

                try:
                    var_val = float(val_dir.name)
                except:
                    var_val = val_dir.name

                results = {}

                for result in val_dir.glob(glob):
                    with result.open("r") as fp:
                        res_dict = json.load(fp)

                    for key, val in res_dict.items():
                        if key not in results:
                            results[key] = []
                        results[key].append(val)

                if len(results) == 0:
                    continue

                empty = False

                var_dict["values"].append(var_val)
                var_dict["N"] = []

                for key, val in results.items():
                    mean = np.mean(val, 0).tolist()
                    std = np.std(val, 0).tolist()

                    var_dict["N"].append(len(val))

                    if key not in var_dict:
                        var_dict[key] = []

                    var_dict[key].append((mean, std))

            if not empty:
                l1_dict["values"].append(l1_val)
                l1_dict["results"].append(var_dict)

    arrays = {}
    for key in l1_dict["results"][0]:
        if key not in ("variable", "values"):
            arrays[key] = []

    l1_lam = l1_dict["values"]

    for res_dict in l1_dict["results"]:
        l2_lam = res_dict["values"]

        for key in arrays:
            arrays[key].append(res_dict[key])

    for elmt in arrays:
        arrays[elmt] = np.array(arrays[elmt])

    arrays["l1_lam"] = np.array(l1_lam)
    arrays["l2_lam"] = np.array(l2_lam)

    out_file = root_path / "lsurface.npz"
    np.savez(out_file, **arrays)
