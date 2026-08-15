#!/usr/bin/env python3
"""Run one baseline script on a core7 panel with the target columns excluded.

The core7 panels carry three TARGET columns (RET_CS, RET_CS_Z, RET_CS5) in the
same per-stock CSV header as the seven features.  RET_CS5 is FORWARD-LOOKING
(prep_panel.py builds it from rows i+1..i+5), so it may never be a model
input; RET_CS / RET_CS_Z are excluded as well so the input set is exactly the
pre-registered core7 feature list (RET_CS_IN is the sanctioned duplicate that
carries yesterday's CS return as an input).  The baselines package
(scripts/pooled/baselines/) is header-driven and has no exclusion flag, and
this project does not modify it, so this wrapper patches panel.Panel AFTER
construction -- the same post-load column drop the controls package performs
in controls/common.py apply_feature_selection -- and then executes the
baseline script unchanged via runpy.

    python3 run_one.py trivial.py --panel .../set1_core7 --param RET_CS \
        --model str1 --out-dir results_econ/str1/set1_core7_seed42

Everything after the script name is passed to the baseline verbatim.
"""

import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINES = os.path.join(os.path.dirname(HERE), "baselines")

EXCLUDE = ("RET_CS", "RET_CS_Z", "RET_CS5")


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    script = sys.argv[1]
    if not os.path.isabs(script):
        script = os.path.join(BASELINES, script)
    if not os.path.exists(script):
        raise SystemExit(f"no such baseline script: {script}")

    sys.path.insert(0, BASELINES)
    import panel as panel_mod  # noqa: E402

    orig_init = panel_mod.Panel.__init__

    def patched_init(self, path, param="RET", score_param=None, realized="auto"):
        # Panel.__init__ extracts Y (the training target) and Yscore (the
        # sidecar's raw realised returns) BEFORE we drop the target columns
        # from the feature matrix, so training on RET_CS still works while no
        # model ever sees RET_CS/RET_CS_Z/RET_CS5 as an input.
        orig_init(self, path, param, score_param, realized)
        drop = [c for c in EXCLUDE if c in self.features]
        if drop:
            keep = [i for i, f in enumerate(self.features) if f not in drop]
            self.features = [self.features[i] for i in keep]
            self.X = self.X[:, :, keep]
            print(f"# econ/run_one: excluded target columns {drop}; "
                  f"model inputs = {self.features}")
        if "RET_CS5" in self.features:
            raise SystemExit("RET_CS5 (forward-looking) survived the feature "
                             "exclusion; refusing to run")

    panel_mod.Panel.__init__ = patched_init

    sys.argv = [script] + sys.argv[2:]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
