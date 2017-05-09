#!/bin/sh

./toyModel/dumpModelPlots.py && mkdir modelPlots && mv *.eps modelPlots/

./toyModel/closureTest.py && mkdir results_withoutSyst && mv *.eps results_withoutSyst/

./toyModel/unfoldWithSyst.py withSyst nobias inputA && mkdir results_unfA_withSyst && mv *.eps results_unfA_withSyst/
./toyModel/unfoldWithSyst.py withSyst nobias inputB && mkdir results_unfB_withSyst && mv *.eps results_unfB_withSyst/
./toyModel/unfoldWithSyst.py withSyst nobias inputC && mkdir results_unfC_withSyst && mv *.eps results_unfC_withSyst/

./toyModel/testBias.py withoutSyst nobias && mkdir results_withoutSyst_regFirstDer && mv *.eps results_withoutSyst_regFirstDer/
./toyModel/testBias.py withoutSyst bias && mkdir results_withoutSyst_regFirstDer_bias && mv *.eps results_withoutSyst_regFirstDer_bias/
./toyModel/testBias.py withSyst nobias && mkdir results_withSyst_regFirstDer && mv *.eps results_withSyst_regFirstDer/
./toyModel/testBias.py withSyst bias && mkdir results_withSyst_regFirstDer_bias && mv *.eps results_withSyst_regFirstDer_bias/

./toyModel/unfoldWithSyst.py withSyst bias inputA && mkdir results_unfA_withSyst_regFirstDer_bias && mv *.eps results_unfA_withSyst_regFirstDer_bias/
./toyModel/unfoldWithSyst.py withSyst bias inputB && mkdir results_unfB_withSyst_regFirstDer_bias && mv *.eps results_unfB_withSyst_regFirstDer_bias/
./toyModel/unfoldWithSyst.py withSyst bias inputC && mkdir results_unfC_withSyst_regFirstDer_bias && mv *.eps results_unfC_withSyst_regFirstDer_bias/

./toyModel/testBiasTUnfold.py nobias && mkdir results_TUnfold_regFirstDer && mv *.eps results_TUnfold_regFirstDer/
./toyModel/testBiasTUnfold.py bias && mkdir results_TUnfold_regFirstDer_bias && mv *.eps results_TUnfold_regFirstDer_bias/

./toyModel/testBiasDagostini.py && mkdir results_Dagostini && mv *.eps results_Dagostini/
