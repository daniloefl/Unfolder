#!/bin/sh

mkdir modelPlots
mkdir results_withoutSyst
mkdir results_unfA_withSyst
mkdir results_unfB_withSyst
mkdir results_unfC_withSyst
mkdir results_unfD_withSyst

./toyModel/dumpModelPlots.py && mv *.eps modelPlots/

./toyModel/unfoldWithSyst.py withSyst nobias inputA && mv *.eps results_unfA_withSyst/
./toyModel/unfoldWithSyst.py withSyst nobias inputB && mv *.eps results_unfB_withSyst/
./toyModel/unfoldWithSyst.py withSyst nobias inputC && mv *.eps results_unfC_withSyst/
./toyModel/unfoldWithSyst.py withSyst nobias inputD && mv *.eps results_unfD_withSyst/

#mkdir results_withoutSyst_regFirstDer
#mkdir results_withoutSyst_regFirstDer_bias
#mkdir results_withSyst_regFirstDer
#mkdir results_withSyst_regFirstDer_bias
#mkdir results_TUnfold_regFirstDer
#mkdir results_TUnfold_regFirstDer_bias
#mkdir results_Dagostini

#./toyModel/testBias.py withSyst nobias && mv *.eps results_withSyst_regFirstDer/
#./toyModel/testBias.py withoutSyst nobias && mv *.eps results_withoutSyst_regFirstDer/
#./toyModel/testBias.py withSyst bias && mv *.eps results_withSyst_regFirstDer_bias/
#./toyModel/testBias.py withoutSyst bias && mv *.eps results_withoutSyst_regFirstDer_bias/

#./toyModel/closureTest.py && mv *.eps results_withoutSyst/

#./toyModel/testBiasTUnfold.py nobias && mv *.eps results_TUnfold_regFirstDer/
#./toyModel/testBiasTUnfold.py bias && mv *.eps results_TUnfold_regFirstDer_bias/

#./toyModel/testBiasDagostini.py && mv *.eps results_Dagostini/


