#!/bin/sh

mkdir modelPlots
mkdir results_withoutSyst
mkdir results_unfA_withSyst_nonlinear
mkdir results_unfB_withSyst_nonlinear

./toyModel/dumpModelPlots.py && mv *.eps modelPlots/
tar cvfz model_toy.tar.gz modelPlots

./toyModel/closureTest.py && mv *.eps results_withoutSyst/
tar cvfz closure_toy.tar.gz results_withoutSyst

./toyModel/unfoldWithSyst.py withSyst nobias inputA && mv *.eps results_unfA_withSyst_nonlinear/
tar cvfz closure_systA_toy.tar.gz results_unfA_withSyst_nonlinear
./toyModel/unfoldWithSyst.py withSyst nobias inputB && mv *.eps results_unfB_withSyst_nonlinear/
tar cvfz closure_systB_toy.tar.gz results_unfB_withSyst_nonlinear

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

#./toyModel/testBiasTUnfold.py nobias && mv *.eps results_TUnfold_regFirstDer/
#./toyModel/testBiasTUnfold.py bias && mv *.eps results_TUnfold_regFirstDer_bias/

#./toyModel/testBiasDagostini.py && mv *.eps results_Dagostini/


