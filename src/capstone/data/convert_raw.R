library(tidyverse)

load('TEP_Faulty_Training.RData')
load('TEP_Faulty_Testing.RData')
load('TEP_FaultFree_Training.RData')
load('TEP_FaultFree_Testing.RData')
 
write_csv(x=fault_free_testing,'fault_free_testing.csv')
write_csv(x=fault_free_training,'fault_free_training.csv')
write_csv(x=faulty_testing,'faulty_testing.csv')
write_csv(x=faulty_training,'faulty_training.csv')