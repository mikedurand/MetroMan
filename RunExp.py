#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from numpy import zeros,reshape,putmask
import pickle
from ReadObs import ReadObs
from ReadParams import ReadParams
from ReadTruth import ReadTruth
from SelObs import SelObs
from CalcdA import CalcdA
from ProcessPrior import ProcessPrior
from GetCovMats import GetCovMats
from MetropolisCalculations import MetropolisCalculations
from CalculateEstimates import CalculateEstimates
from MakeFigs import MakeFigs
from CalcErrorStats import CalcErrorStats
from FilterEstimate import FilterEstimate
from DispRMSEStats import DispRMSEStats

def RunExp(RunDir,ShowFigs,Laterals,ReCalc,DebugMode):
    
    if ReCalc:
        
        print("Running ", RunDir)
    
        #%% read input
        ObsFile= RunDir + '/SWOTObs.txt'
        [DAll,Obs]=ReadObs(ObsFile)
        
        ParamFile= RunDir + '/params.txt'
        [Chain,Prior,R,Exp]=ReadParams(ParamFile,DAll)
        
        TruthFile=RunDir + '/truth.txt'
        AllTruth=ReadTruth(TruthFile,DAll)
        
        if Laterals:
            LateralMeanFile = RunDir + '/LateralsMean.txt'
            from ReadLats import ReadLats   
            Prior.AllLats.q=ReadLats(LateralMeanFile,DAll)     
            Prior.AllLats.qv=reshape(Prior.AllLats.q,(DAll.nR*(DAll.nt-1),1)  )
        else: 
            Prior.AllLats.q=zeros( (DAll.nR,DAll.nt)  )
               
        #%% setup variables
        [D,Obs,AllObs,DAll,Truth,Prior.Lats.q]=SelObs(DAll,Obs,Exp,AllTruth,Prior.AllLats)
        Prior.Lats.qv=reshape(Prior.Lats.q,(D.nR*(D.nt-1),1)  )
        
        Obs=CalcdA(D,Obs)
        AllObs=CalcdA(DAll,AllObs)
        
        [Prior,jmp]=ProcessPrior(Prior,AllObs,DAll,Obs,D,ShowFigs,Exp,R,DebugMode)
        
        [Obs,Prior]=GetCovMats(D,Obs,Prior)
        
        Obs.S[Obs.S<0]=putmask(Obs.S,Obs.S<0,0) #limit slopes to zero
        AllObs.S[AllObs.S<0]=putmask(AllObs.S,AllObs.S<0,0)
        
        #%%
        Chain=MetropolisCalculations(Prior,D,Obs,jmp,Chain,R,DAll,AllObs,Exp.nOpt,DebugMode)
        
        #%%
        filename=RunDir+ '/RunData.pkl'
        with open(filename, 'wb') as output:
            pickle.dump(AllObs, output, pickle.HIGHEST_PROTOCOL)
            pickle.dump(AllTruth,output, pickle.HIGHEST_PROTOCOL)
            pickle.dump(Chain,output,pickle.HIGHEST_PROTOCOL)
            pickle.dump(D,output,pickle.HIGHEST_PROTOCOL)
            pickle.dump(DAll,output,pickle.HIGHEST_PROTOCOL)
            pickle.dump(Exp,output,pickle.HIGHEST_PROTOCOL)     
            pickle.dump(Laterals,output,pickle.HIGHEST_PROTOCOL)                 
            pickle.dump(Obs, output, pickle.HIGHEST_PROTOCOL)
            pickle.dump(Prior, output, pickle.HIGHEST_PROTOCOL)   
            pickle.dump(R, output, pickle.HIGHEST_PROTOCOL)   
            pickle.dump(Truth, output, pickle.HIGHEST_PROTOCOL)    
            pickle.dump(jmp, output, pickle.HIGHEST_PROTOCOL)         
        
    else:
        filename=RunDir+ '/RunData.pkl'
        with open(filename, 'rb') as input:
            AllObs = pickle.load(input)
            AllTruth=pickle.load(input)
            Chain=pickle.load(input)
            D=pickle.load(input)
            DAll=pickle.load(input)
            Exp=pickle.load(input)            
            Laterals=pickle.load(input)                        
            Obs = pickle.load(input)        
            Prior = pickle.load(input)
            R = pickle.load(input)
            Truth = pickle.load(input)
            jmp = pickle.load(input)
    
    [Estimate,Chain]=CalculateEstimates(Chain,D,Obs,Prior,DAll,AllObs,Exp.nOpt)
    Estimate = FilterEstimate(Estimate,Chain,D,Obs)
    
    #%%
    Err=CalcErrorStats(AllTruth,Estimate,DAll)
    Err=DispRMSEStats(Err,Truth,Prior,Estimate)
    
    if ShowFigs:
        MakeFigs(D,Truth,Prior,Chain,Estimate,Err,AllTruth,DAll,AllObs)
    
    filename=RunDir+ '/EstData.pkl'
    with open(filename, 'wb') as output:
        pickle.dump(Estimate, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(Err, output, pickle.HIGHEST_PROTOCOL)