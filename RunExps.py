#!/usr/bin/env python3

from RunExp import RunExp

# fid=open("/Users/mtd/Box Sync/Analysis/SWOT/Discharge/MetroManExperiments/tstFile.txt","r")
fid=open("RunFile.txt","r")
infile=fid.readlines()

ShowFigs=True
Laterals=False
ReCalc=True #setting to False will load the existing .pkl run data and plot


for i in range(0,len(infile)):
    RunDir=infile[i].replace('\n','')
    RunExp(RunDir,ShowFigs,Laterals,ReCalc)
        
